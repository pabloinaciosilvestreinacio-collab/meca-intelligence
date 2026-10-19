'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const Lab=require('../src/core.js');

test('900 unique examples, correct splits and balanced classes',()=>{
 const d=Lab.createDataset();
 assert.deepEqual([d.training.length,d.validation.length,d.testing.length],[600,150,150]);
 const all=[...d.training,...d.validation,...d.testing];
 assert.equal(new Set(all.map(s=>s.sampleId)).size,900);
 for(const s of all){assert.ok(Lab.LABELS.includes(s.label));assert.equal(s.rows.length,40);assert.equal(Lab.flatten(s).length,200);assert.equal(s.origin,'synthetic');}
 for(const k of ['training','validation','testing']){
  const cls=d[k].reduce((o,s)=>(o[s.label]=(o[s.label]||0)+1,o),{});
  assert.equal(cls.imu_travada,d[k].length/2);assert.equal(cls.normal,d[k].length/2);
 }
});
test('deterministic and independent synthetic datasets',()=>{
 const a=Lab.createDataset(32578),b=Lab.createDataset(32578),c=Lab.createDataset(4321);
 assert.deepEqual(a.testing[0],b.testing[0]);assert.notDeepEqual(a.testing[0],c.testing[0]);
});
test('CSV roundtrip preserves channels and 20Hz time',()=>{
 const d=Lab.createDataset(123,{training:2,validation:2,testing:2});
 const example=d.testing[0],csv=Lab.csvForSample(example);const restored=Lab.csvParse(csv,example.label,example.sampleId);
 assert.deepEqual(restored.rows,example.rows);
 assert.equal(Lab.toFeatures(Lab.flatten(restored)).length,200);
 assert.throws(()=>Lab.csvParse(csv.replace('timestamp','time')),/cabeçalho/i);
});
test('browser MLP actually updates parameters and generalizes on held-out synthetic data',async()=>{
 const d=Lab.createDataset(32578,{training:120,validation:40,testing:40});
 const base=Lab.initModel(32578),initial=base.W[0][0][0];
 const m=await Lab.train({dataset:d,epochs:6,seed:32578,yieldTick:async()=>{}});
 assert.notEqual(m.W[0][0][0],initial);
 assert.ok(Lab.evalModel(m,d.testing).accuracy>=.65);
 assert.equal(Lab.infer(m,Lab.flatten(d.testing[0])).length,4);
 const p=Lab.infer(m,Lab.flatten(d.testing[0]))[3];assert.ok(Math.abs(p[0]+p[1]-1)<1e-5);
});
test('ZIP contains valid binary header and both training/testing directories',async()=>{
 const d=Lab.createDataset(25,{training:4,validation:2,testing:2});
 const zip=Lab.datasetZip(d);assert.equal(zip.type,'application/zip');
 const bytes=new Uint8Array(await zip.arrayBuffer());
 assert.deepEqual(Array.from(bytes.slice(0,4)),[80,75,3,4]);
 const chars=new TextDecoder().decode(bytes);
 assert.ok(chars.includes('training/'));assert.ok(chars.includes('testing/'));assert.ok(chars.includes('MANIFESTO.txt'));
});