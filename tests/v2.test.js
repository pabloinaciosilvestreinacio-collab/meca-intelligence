'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
require('../src/core.js');
const V=require('../src/v2-core.js');
test('four architectures initialize, forward, gradient and JSON roundtrip',()=>{
 for(const hidden of [[20,10],[32,16],[64,32,16],[128,64,32]]){
  const m=V.init({hidden},12),raw=Array(200).fill(.25);
  assert.deepEqual(m.shape,[200,...hidden,2]);
  assert.equal(m.W.length,hidden.length+1);
  assert.equal(V.parameters(m.shape),m.W.flatMap(l=>l.map(row=>row.length+1)).reduce((a,b)=>a+b,0));
  const p=V.forward(m,raw).at(-1);assert.ok(Math.abs(p[0]+p[1]-1)<1e-12);
  const saved=V.legacy(JSON.parse(JSON.stringify(m)));
  assert.deepEqual(V.forward(saved,raw).at(-1),p);
 }
});
test('legacy model 200/20/10/2 imports and matches infer',()=>{
 const core=require('../src/core.js'),m=core.initModel(32578),raw=core.flatten(core.synth(5,'normal','curva'));
 const migrated=V.legacy(m);assert.equal(migrated.origin,'legacy_browser_import');
 for(let i=0;i<2;i++)assert.ok(Math.abs(core.infer(m,raw)[3][i]-V.forward(migrated,raw).at(-1)[i])<1e-5);
});
test('deterministic data and multiple sizes',()=>{
 const a=V.samples(1000,32578),b=V.samples(1000,32578),c=V.samples(1000,100);
 assert.equal(a.training.length+a.validation.length+a.testing.length,1000);
 assert.deepEqual(a.testing[0],b.testing[0]);
 assert.notDeepEqual(a.training[0],c.training[0]);
 assert.throws(()=>V.samples(26000,2),/Amostras/);
});
test('finite difference gradient check on a small dense network',()=>{
 const m=V.init({hidden:[2],activation:'tanh'},72),x=Float64Array.from({length:200},(_,i)=>(i%9-4)/100);
 const analytic=V.gradients(m,x,1),eps=1e-5;
 for(const [l,j,i] of [[0,0,0],[0,1,17],[1,1,0]]){
  const previous=m.W[l][j][i];
  m.W[l][j][i]=previous+eps;const up=V.gradients(m,x,1).loss;
  m.W[l][j][i]=previous-eps;const down=V.gradients(m,x,1).loss;
  m.W[l][j][i]=previous;
  assert.ok(Math.abs((up-down)/(2*eps)-analytic.gw[l][j][i])<1e-6);
 }
});
test('train updates weights, finite loss, exports model and holds independent synthetic test set',async()=>{
 const ds=V.samples(120,32578),m=V.init({hidden:[8,4]},6),w=m.W[0][0][0];
 const r=await V.train(m,ds,{epochs:2,batchSize:16,lr:.003,seed:6,yieldTick:async()=>{}});
 assert.equal(r.cancelled,false);
 assert.notEqual(JSON.stringify(r.model.W),JSON.stringify(m.W));
 assert.ok(r.history.every(x=>Number.isFinite(x.validation.loss)));
 assert.equal(V.evaluate(r.model,ds.testing).total,ds.testing.length);
 assert.ok(r.model.trained.history.length===2);
 assert.deepEqual(V.forward(r.model,Array(200).fill(0)).at(-1),V.forward(V.legacy(JSON.parse(JSON.stringify(r.model))),Array(200).fill(0)).at(-1));
});
test('cancel is safe and stops without substituting a pretend trained model',async()=>{
 const ds=V.samples(80,10),m=V.init({hidden:[6]},1);
 const result=await V.train(m,ds,{epochs:8,shouldStop:()=>true,yieldTick:async()=>{}});
 assert.equal(result.cancelled,true);assert.equal(m.trained,null);
});
test('invalid dimensions, labels and values rejected',()=>{
 assert.throws(()=>V.config({hidden:[0]}));
 assert.throws(()=>V.config({hidden:[128,128,128,128]}),/60.000/);
 assert.throws(()=>V.forward(V.init({hidden:[2]}),[1,2]));
 assert.throws(()=>V.legacy({kind:'browser',shape:[200,2],W:[[[0]]],B:[[0,0]]}));
});
test('MIH-V1 file validates channels and does not synthesize IMU readings',()=>{
 const good={
  schema_version:'MIH-V1',source:'real',samples:Array.from({length:40},(_,i)=>({
   time_ms:i*50,imu_yaw_deg:i,encoder_md_ticks:i*10,encoder_me_ticks:i*10,
   motor_md_command:.4,motor_me_command:.4
  }))
 };
 const imp=V.importTelemetry(JSON.stringify(good),'telemetry.json','real');
 assert.equal(imp.compatible,true);assert.equal(imp.raw.length,200);
 const missing=JSON.parse(JSON.stringify(good));delete missing.samples[2].imu_yaw_deg;
 assert.equal(V.importTelemetry(JSON.stringify(missing),'telemetry.json','real').compatible,false);
 assert.throws(()=>V.importTelemetry(JSON.stringify(good),'telemetry.json','simulation'),/origem/);
 const sim=V.importTelemetry('time_s,x_m,y_m,yaw_deg,ME_ticks,MD_ticks\n0,1,1,0,0,0\n.05,1,1,0,10,10','trajectory.csv','simulation');
 assert.equal(sim.compatible,false);assert.ok(sim.missing.includes('imu_yaw_deg'));
});
test('input format, optimizers and early stop metadata',async()=>{
 const data=V.samples(70,10),m=V.init({hidden:[6],activation:'relu'},10);
 const r=await V.train(m,data,{epochs:1,batchSize:8,optimizer:'sgd',lr:.001,yieldTick:async()=>{}});
 assert.equal(r.model.trained.options.optimizer,'sgd');assert.equal(r.history.length,1);
 assert.throws(()=>V.trainingOptions({optimizer:'magical'}));
});