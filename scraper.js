const fs = require("fs");

const puppeteer = require("puppeteer");

(async () => {
    const pLimit = (await import("p-limit")).default;

    const limit = pLimit(16);
    const browser = await puppeteer.launch();

    const scores = {};

    async function getScore(size, id) {
        console.log("Fetching score for", size + id);
        const page = await browser.newPage();
        await page.goto(
            `https://170-leaderboard.vercel.app/input/${size}/${id}`
        );
        await page.waitForSelector("td:nth-child(3)");
        const score = await page.evaluate(() => {
            return parseFloat(
                document.querySelector("td:nth-child(3)").innerText
            );
        });
        scores[size + id] = score;
        await page.close();
    }

    const promises = [];
    for (const id of [...Array(260).keys()].map((i) => i + 1)) {
        for (const size of ["small", "medium", "large"]) {
            promises.push(limit(() => getScore(size, id)));
        }
    }
    await Promise.all(promises);
    console.log(scores);
    const sorted = {};
    for (const size of ["small", "medium", "large"]) {
        for (const id of [...Array(260).keys()].map((i) => i + 1)) {
            sorted[size + id] = scores[size + id];
        }
    }
    console.log(sorted);
    fs.writeFileSync("scores.json", JSON.stringify(sorted, null, 2));
    fs.writeFileSync("scores.txt", Object.values(sorted).join("\n") + "\n");
    process.exit(0);
})();
