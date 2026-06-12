import asyncio
import sys

from playwright.async_api import async_playwright


async def main():
    wait_s = float(sys.argv[1]) if len(sys.argv) > 1 else 25
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        await page.goto("http://localhost:5179/")
        await page.wait_for_timeout(3000)
        await page.screenshot(path="test/shot_loading.png")
        # wait for loader to finish (model load) then for generation to begin
        try:
            await page.wait_for_selector("#loader.ready", timeout=120000)
            await page.click("#enter-muted")
            await page.wait_for_selector("#loader.done")
        except Exception:
            errors.append("LOADER NEVER FINISHED")
        await page.wait_for_timeout(wait_s * 1000)
        await page.screenshot(path="test/shot_running.png")
        await page.wait_for_timeout(1100)
        await page.screenshot(path="test/shot_b.png")
        transcript = await page.inner_text("#transcript-inner")
        print("TRANSCRIPT:", transcript[:500])
        print("TOKENS:", await page.inner_text("#tok-count"))
        print("ERRORS:")
        for e in errors[:30]:
            print(" -", e[:300])
        await browser.close()


asyncio.run(main())
