/* Real Chromium E2E. Tests run under GitHub Actions, screenshot artifacts included. */
'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const root=path.resolve(__dirname,'..');
let server,browser,origin,ctx,page;
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.jpg':'image/jpeg'};
test.before(async()=>{
 server=http.createServer((req,res)=>{
  const name=decodeURIComponent((req.url||'/').split('?')[0])||'/';
  const filename=path.resolve(root,'.'+(name==='/'?'/index.html':name));
  if(!filename.startsWith(root+path.sep)){res.writeHead(403).end('Forbidden');return;}
  fs.readFile(filename,(err,b)=>{if(err){res.writeHead(404).end('Not found');return;}
   res.writeHead(200,{'Content-Type':types[path.extname(filename)]||'text/plain'}).end(b);});
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 origin='http://127.0.0.1:'+server.address().port;
 browser=await chromium.launch({headless:true});
 ctx=await browser.newContext({acceptDownloads:true,viewport:{width:1400,height:850}});
 page=await ctx.newPage();
});
test.after(async()=>{
 await ctx?.close();await browser?.close();await new Promise(resolve=>server.close(resolve));
});
test('dashboard, dynamic network, worker training, test and real activations',async()=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(origin,{waitUntil:'load'});
 await page.locator('#v2Init').click();
 await page.waitForFunction(()=>document.querySelector('#v2Architecture')?.textContent.includes('200 → 20 → 10 → 2'));
 await page.selectOption('#v2Preset','32,16');
 await page.locator('#v2Init').click();
 await page.waitForFunction(()=>document.querySelector('#v2Architecture')?.textContent.includes('200 → 32 → 16 → 2'));
 assert.match(await page.locator('#v2GraphStats').innerText(),/neurônios visíveis/);
 await page.selectOption('#v2Count','100');
 await page.locator('#v2Epochs').fill('1');
 await page.locator('#v2Generate').click();
 await page.waitForFunction(()=>document.querySelector('#v2Counts')?.textContent.includes('training: 70'));
 await page.locator('#v2Train').click();
 await page.waitForFunction(()=>document.querySelector('#v2ModelType')?.textContent.includes('Treinado'),{timeout:60000});
 const before=await page.locator('#v2TrainMeta').innerText();
 assert.match(before,/Época 1/);
 await page.locator('#v2Test').click();
 await page.waitForFunction(()=>document.querySelector('#v2Results')?.textContent.includes('accuracy'));
 await page.locator('#v2Sample').click();
 await page.waitForFunction(()=>document.querySelector('#v2SampleInfo')?.textContent.includes('predição'));
 await page.locator('#v2Density').fill('30');
 await page.locator('#v2Radius').fill('7');
 await page.locator('#v2Canvas').click({position:{x:60,y:50}});
 const graph=await page.locator('#v2GraphStats').innerText();
 const matched=graph.match(/(\d+)\/\d+ conexões visíveis/);
 assert.ok(matched&&Number(matched[1])>0,'Selected neuron must reveal actual incident weights: '+graph);
 await page.locator('#v2ResetStyle').click();
 await page.screenshot({path:'v2-desktop.png',fullPage:true});
 assert.deepEqual(errors,[]);
});
test('download and reload V2 JSON model without altering predictions',async()=>{
 const original=await page.evaluate(()=>{
  const v=window.MecaV2; const ds=v.samples(30,3);return v.forward(v.init({hidden:[32,16]},32578),v.sampleRaw(ds.testing[0])).at(-1);
 });
 const [reportDownload]=await Promise.all([
  page.waitForEvent('download'),page.locator('#v2Report').click(),
 ]);
 const report=JSON.parse(fs.readFileSync(await reportDownload.path(),'utf8'));
 assert.equal(report.schema,'MIH-V2-EXPERIMENT-v1');
 assert.deepEqual(report.architecture.shape,[200,32,16,2]);
 assert.equal(report.dataset.baselineOnValidation.matrix.length,2);
 assert.equal(report.evaluation.total,15);
 assert.equal(report.imported_file,null);
 assert.match(report.limits.join(' '),/Mequinha físico/);
 const [download]=await Promise.all([
  page.waitForEvent('download'),page.locator('#v2Save').click(),
 ]);
 const filename=await download.path();const json=JSON.parse(fs.readFileSync(filename,'utf8'));
 assert.deepEqual(json.shape,[200,32,16,2]);assert.ok(json.trained);
 await page.once('dialog',dialog=>dialog.accept());
 await page.locator('#v2ModelFile').setInputFiles({
  name:'saved-v2.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(json))});
 await page.waitForFunction(()=>document.querySelector('#v2ModelType')?.textContent.includes('Treinado'));
 assert.match(await page.locator('#v2Architecture').innerText(),/200 → 32 → 16 → 2/);
 const alternative=await page.evaluate(()=>window.MecaV2.init({hidden:[20,10]},7));
 await page.locator('#v2CompareFiles').setInputFiles([
  {name:'A.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(json))},
  {name:'B.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(alternative))}
 ]);
 await page.locator('#v2Compare').click();
 await page.waitForFunction(()=>document.querySelector('#v2Comparison')?.textContent.includes('"results"'));
 assert.match(await page.locator('#v2Comparison').innerText(),/accuracy/);
 const pred=await page.evaluate(model=>{
  const V=window.MecaV2,ds=V.samples(30,3),x=V.sampleRaw(ds.testing[0]);
  const a=V.forward(model,x).at(-1),b=V.forward(V.legacy(JSON.parse(JSON.stringify(model))),x).at(-1);
  return {a,b};
 },json);
 assert.deepEqual(pred.a,pred.b);
 const empty=await page.evaluate(()=>window.MecaV2.init({hidden:[2]}));
 assert.notDeepEqual(pred.a,await page.evaluate(m=>window.MecaV2.forward(m,window.MecaV2.sampleRaw(window.MecaV2.samples(30,3).testing[0])).at(-1),empty));
});
test('telemetry importer parses valid JSON, rejects missing sensor and shows preview',async()=>{
 const data={schema_version:'MIH-V1',source:'real',run_id:'real-001',timestamp:'2026-10-19T20:00:00-03:00',robot_config_id:'bench',code_version:'test',experiment_id:'test-e2e',samples:Array.from({length:40},(_,i)=>({
  time_ms:i*50,imu_yaw_deg:i*.1,encoder_md_ticks:i*15,encoder_me_ticks:i*14,
  motor_md_command:.4,motor_me_command:.4
 }))};
 await page.locator('#v2Origin').selectOption('real');
 await page.locator('#v2ImportFile').setInputFiles({
  name:'mequinha.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});
 await page.waitForFunction(()=>document.querySelector('#v2ImportResult')?.textContent.includes('"compatible": true'));
 assert.equal(await page.locator('#v2InferFile').isDisabled(),false);
 await page.locator('#v2InferFile').click();
 assert.match(await page.locator('#v2ImportPrediction').innerText(),/PREDIÇÃO EXPERIMENTAL/);
 delete data.samples[4].imu_yaw_deg;
 await page.locator('#v2ImportFile').setInputFiles({
  name:'mequinha-invalid.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});
 await page.waitForFunction(()=>document.querySelector('#v2ImportResult')?.textContent.includes('"compatible": false'));
 assert.equal(await page.locator('#v2InferFile').isDisabled(),true);
});
test('a real Worker job can be cancelled without exporting a fake result',async()=>{
 const cancelPage=await ctx.newPage();
 try{
  await cancelPage.goto(origin);
  await cancelPage.locator('#v2Init').click();
  await cancelPage.waitForFunction(()=>document.querySelector('#v2Architecture')?.textContent.includes('200 → 20 → 10 → 2'));
  await cancelPage.locator('#v2Generate').click();
  await cancelPage.waitForFunction(()=>document.querySelector('#v2Counts')?.textContent.includes('training: 700'));
  await cancelPage.locator('#v2Epochs').fill('40');
  await cancelPage.locator('#v2Train').click();
  await cancelPage.waitForFunction(()=>document.querySelector('#v2TrainMeta')?.textContent.includes('Época 1'),null,{timeout:45000});
  await cancelPage.locator('#v2Cancel').click();
  await cancelPage.waitForFunction(()=>/cancelado|Cancelamento solicitado/.test(document.querySelector('#v2Status')?.textContent||''),null,{timeout:30000});
  assert.match(await cancelPage.locator('#v2Status').innerText(),/cancelado|Cancelamento solicitado/);
 }finally{await cancelPage.close();}
});
test('real mobile viewport remains usable, and no false GPU/VM state',async()=>{
 const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 const mp=await mobile.newPage();await mp.goto(origin);
 assert.equal(await mp.locator('#neuralV2').isVisible(),true);
 const env=await mp.locator('#v2Environment').innerText();
 assert.match(env,/CPU local|Indisponível/);
 assert.doesNotMatch(env,/GPU disponível|VM conectada/);
 await mp.screenshot({path:'v2-mobile.png',fullPage:true});
 await mobile.close();
});