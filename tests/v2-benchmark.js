/* Actual measurements, no estimates passed off as elapsed time.
 * CI times are runner measurements, not user hardware benchmarks.
 */
'use strict';
require('../src/core.js');
const V=require('../src/v2-core.js');
const {performance}=require('node:perf_hooks');
(async()=>{
 const results=[];
 for(const count of [1000,5000,10000]){
  const t=performance.now(),ds=V.samples(count,32578),generationMs=performance.now()-t;
  const model=V.init({hidden:[20,10]},32578),a=performance.now();
  const trained=await V.train(model,ds,{epochs:1,batchSize:64,lr:.003,yieldTick:async()=>{}});
  results.push({count,generationMs:Math.round(generationMs),
    trainingMs:Math.round(performance.now()-a),
    env:'GitHub Actions Linux Node CPU',node:process.version,platform:process.platform,
    arch:process.arch,epochs:1,batchSize:64,
    heapUsedMiB:Math.round(process.memoryUsage().heapUsed/1048576*100)/100,
    success:!trained.cancelled,finalValidationLoss:trained.history.at(-1)?.validation.loss});
 }
 console.log(JSON.stringify({schema:'MIH-V2-BENCHMARK',timestamp:new Date().toISOString(),results},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});