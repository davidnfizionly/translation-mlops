// Headless end-to-end run of the translation widget (automated captures).
// Requires the dev server on 127.0.0.1:5173. Usage: node e2e/flow.mjs <screenshot-dir> [offline|live]
// offline: the endpoint does not exist; checks stored outputs, the refusal of free text and the zero cost state.
// live: the endpoint is InService; sends the demo sentence and the 10 stored test sentences.
// Playwright is resolved from PLAYWRIGHT_MODULE (path to an installed playwright package) or the default resolution.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');

const out = process.argv[2] ?? '.';
const mode = process.argv[3] ?? 'offline';
const log = (msg) => console.log(`${new Date().toISOString()} ${msg}`);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const section = (title) => page.locator('section', { has: page.getByRole('heading', { name: title }) });
const result = page.getByTestId('result');

await page.goto('http://127.0.0.1:5173/');
await page.getByText('Tajnaam').first().waitFor();
await page.getByText('very probably machine generated').first().waitFor();
await page.getByText('not reliable translations').waitFor();
await page.getByText('The fine-tuned model is not published').waitFor();
log('attribution, licenses and reference warning shown');

await page.getByRole('cell', { name: '35.51' }).waitFor();
await page.getByRole('cell', { name: '68.37' }).waitFor();
log('stored metrics shown');

if (mode === 'offline') {
  await page.getByText('Endpoint not deployed: stored outputs only').waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Translate', exact: true }).click();
  await result.getByText('Endpoint not deployed').waitFor({ timeout: 30000 });
  log('free text refused without an endpoint');
  await page.getByRole('button', { name: 'English to Nuer' }).first().click();
  await page.getByRole('button', { name: 'Send the 10 stored test sentences' }).click();
  await page.getByTestId('batch').getByText('Sent 10 of 10: 10 answered from stored outputs').waitFor({ timeout: 30000 });
  log('10 stored Nuer sentences answered from stored outputs');
  await page.getByText('not deployed: 0 USD per hour now').waitFor();
  log('cost panel shows no endpoint cost');
} else {
  await page.getByText('Endpoint InService').waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Boil water before drinking.' }).click();
  await page.getByRole('button', { name: 'Translate', exact: true }).click();
  await result.getByText('live endpoint').waitFor({ timeout: 120000 });
  await section('Live translation').screenshot({ path: `${out}/auto-01-widget-live-translation.png` });
  log(`live Dinka output: ${await result.locator('p').nth(1).innerText()}`);
  await page.getByRole('button', { name: 'Send the 10 stored test sentences' }).click();
  await page.getByTestId('batch').getByText('Sent 10 of 10: 10 answered by the live endpoint').waitFor({ timeout: 300000 });
  log('10 stored Dinka sentences answered by the live endpoint');
  await page.waitForTimeout(6000);
  await section('Estimated cost (estimate)').screenshot({ path: `${out}/auto-02-widget-cost-counter.png` });
  log(`requests counted: ${await page.getByTestId('Live requests').innerText()}`);
}

await section('Before and after fine-tuning (stored evaluation)').screenshot({ path: `${out}/auto-03-widget-before-after.png` });
await page.screenshot({ path: `${out}/auto-00-widget-full-${mode}.png`, fullPage: true });
log('screenshots written');
await browser.close();
