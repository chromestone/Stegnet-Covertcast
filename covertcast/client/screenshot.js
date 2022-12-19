// Source:
// https://superuser.com/questions/1209741/how-to-take-a-screenshot-of-a-page-n-seconds-after-page-is-loaded-with-chrome-he

const puppeteer = require('puppeteer');

function timeout(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
};

(async() => {
	const browser = await puppeteer.launch();
	const page = await browser.newPage();
	await page.goto('https://www.youtube.com/watch?v=w_Ma8oQLmSM');
	await timeout(10000)
	await page.screenshot({path: 'example.png'});
	browser.close();
})();
