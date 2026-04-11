"""
Browse tweets from a Twitter/X profile using the logged-in Chrome profile.

Copies your Chrome profile to a temp directory (same approach as browser-use)
so Chrome can stay open while this script runs — no singleton lock conflict.

Usage:
    uv run python browse_tweets.py --user Alice --count 10
"""

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

CHROME_USER_DATA_DIR = Path.home() / "Library/Application Support/Google/Chrome"
CHROME_PROFILE = "Default"


def copy_profile_to_temp() -> str:
    """Copy Chrome profile to a temp dir (mirrors browser-use's _copy_profile).
    This avoids the SingletonLock conflict when Chrome is already running."""
    tmp = tempfile.mkdtemp(prefix="browse-tweets-profile-")
    src_profile = CHROME_USER_DATA_DIR / CHROME_PROFILE
    dst_profile = Path(tmp) / CHROME_PROFILE

    if src_profile.exists():
        shutil.copytree(src_profile, dst_profile)
        # Local State is needed for Chrome to recognise the profile
        local_state_src = CHROME_USER_DATA_DIR / "Local State"
        if local_state_src.exists():
            shutil.copy(local_state_src, Path(tmp) / "Local State")

    print(f"Profile copied to temp dir: {tmp}")
    return tmp


async def fetch_tweets(username: str, count: int = 10) -> list[dict]:
    tmp_profile = copy_profile_to_temp()
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=tmp_profile,
            channel="chrome",  # uses installed Chrome binary
            headless=False,  # Set to True once you've confirmed it works
            args=["--disable-blink-features=AutomationControlled"],
            # Playwright injects --use-mock-keychain by default, which prevents
            # Chrome from decrypting cookies stored with the macOS Keychain.
            # Removing it lets the real Keychain be used → cookies work.
            ignore_default_args=["--use-mock-keychain"],
        )

        page = context.pages[0] if context.pages else await context.new_page()

        print(f"Navigating to https://x.com/{username} ...")
        await page.goto(f"https://x.com/{username}", wait_until="domcontentloaded")

        # Wait for the first tweet to appear
        try:
            await page.wait_for_selector('article[data-testid="tweet"]', timeout=20000)
        except Exception:
            print(f"Page title: {await page.title()}")
            print(f"Page URL:   {page.url}")
            await page.screenshot(path="debug_screenshot.png")
            print("Screenshot saved to debug_screenshot.png")
            print("ERROR: No tweets found. Are you logged in? Is the account public?")
            await context.close()
            return []

        tweets: list[dict] = []
        seen_texts: set[str] = set()

        # Scroll and collect until we have enough
        scroll_attempts = 0
        max_scroll_attempts = 20

        while len(tweets) < count and scroll_attempts < max_scroll_attempts:
            articles = await page.query_selector_all('article[data-testid="tweet"]')

            for article in articles:
                if len(tweets) >= count:
                    break

                # --- Tweet text ---
                text_el = await article.query_selector('[data-testid="tweetText"]')
                text = (await text_el.inner_text()).strip() if text_el else ""

                # Skip duplicates and empty tweets (e.g. media-only)
                dedup_key = text[:80]
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)

                # --- Like count ---
                # The like button has aria-label like "123 Likes" or just the count in a span
                likes = ""
                like_btn = await article.query_selector('[data-testid="like"]')
                if like_btn:
                    aria = await like_btn.get_attribute("aria-label") or ""
                    if aria:
                        # e.g. "1234 Likes" → "1234"
                        likes = aria.split()[0]
                    else:
                        # Fallback: read inner span text
                        span = await like_btn.query_selector("span[data-testid='app-text-transition-container']")
                        if span:
                            likes = (await span.inner_text()).strip()

                # --- View count ---
                # The analytics link aria-label is exactly "N views" (e.g. "42094 views")
                views = ""
                analytics = await article.query_selector('a[href*="/analytics"]')
                if analytics:
                    aria = await analytics.get_attribute("aria-label") or ""
                    # aria-label is "42,095. View post analytics" → take the number before "."
                    views = aria.split(".")[0].replace(",", "").replace(" views", "").strip()

                tweets.append({"text": text, "likes": likes, "views": views})
                print(f"  [{len(tweets)}/{count}] {text[:60]!r}  likes={likes}  views={views}")

            if len(tweets) < count:
                # Scroll down to load more tweets
                await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                await page.wait_for_timeout(1500)
                scroll_attempts += 1

        await context.close()
        return tweets[:count]


def main():
    parser = argparse.ArgumentParser(description="Fetch tweets from an X/Twitter profile")
    parser.add_argument("--user", required=True, help="Twitter username (without @)")
    parser.add_argument("--count", type=int, default=10, help="Number of tweets to fetch")
    args = parser.parse_args()

    results = asyncio.run(fetch_tweets(args.user, args.count))

    print("\n--- Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
