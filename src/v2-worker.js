/* Web Worker CPU: does not communicate with a server or access hardware. */
'use strict';
importScripts('./core.js','./v2-core.js');
let data=null,model=null,cancelRequested=false,busy=false;
const send=(event,payload={})=>postMessage({event,...payload});
onmessage=async ({data:req})=>{
 try{
  if(req.action==='cancel'){cancelRequested=true;send('cancelling');return;}
  if(busy)throw Error('Já existe uma tarefa em execução.');
  if(req.action==='generate'){
   busy=true;const started=performance.now();
   const cfg=req.options,estimate=MecaV2.estimate(cfg.count,cfg.shape,cfg.epochs);
   if(cfg.count>10000)throw Error('Mais de 10.000 amostras aguardam benchmark local. Selecione lote menor.');
   data=MecaV2.samples(cfg.count,cfg.seed,cfg.split);
   send('generated',{counts:Object.fromEntries(['training','validation','testing'].map(k=>[k,data[k].length])),
    metadata:data.metadata,estimate,generationMs:Math.round(performance.now()-started)});
  }else if(req.action==='initialize'){
   model=MecaV2.init(req.options,req.seed);send('initialized',{model});
  }else if(req.action==='load'){
   model=MecaV2.legacy(req.model);send('initialized',{model});
  }else if(req.action==='train'){
   if(!data)throw Error('Gere dados antes do treinamento.');
   if(!model)throw Error('Inicialize uma arquitetura primeiro.');
   busy=true;cancelRequested=false;const t=performance.now();
   const result=await MecaV2.train(model,data,{...req.options,
    onEpoch:stat=>send('epoch',{stat,elapsedMs:Math.round(performance.now()-t)}),
    shouldStop:()=>cancelRequested});
   if(result.cancelled)send('cancelled',{epochsCompleted:result.epochsCompleted});
   else{model=result.model;send('trained',{model,history:result.history,
     trainingMs:Math.round(performance.now()-t),environment:'browser-cpu-worker'});}
  }else if(req.action==='test'){
   if(!data||!model)throw Error('Gere dados e treine ou carregue um modelo.');
   send('tested',{result:MecaV2.evaluate(model,data.testing),total:data.testing.length});
  }else if(req.action==='sample'){
   if(!data||!model)throw Error('Gere dados e inicialize um modelo.');
   const index=Math.max(0,Math.min(data.testing.length-1,Math.floor(req.index||0)));
   const sample=data.testing[index],acts=MecaV2.forward(model,MecaV2.sampleRaw(sample));
   send('sample',{index,label:sample.label,scenario:sample.scenario,sampleId:sample.sampleId,acts});
  }else if(req.action==='benchmark'){
   if(busy)throw Error('Processamento ocupado.');
   busy=true;const results=[];const sizes=[1000,5000,10000];
   for(const count of sizes){
    if(cancelRequested)break;
    const tick=performance.now(),ds=MecaV2.samples(count,32578,[.7,.15,.15]),generationMs=performance.now()-tick;
    const network=MecaV2.init({hidden:[20,10]},32578),start=performance.now();
    const out=await MecaV2.train(network,ds,{epochs:1,batchSize:64,lr:.003,
       onEpoch:()=>{},shouldStop:()=>cancelRequested});
    results.push({count,generationMs:Math.round(generationMs),
       trainingMs:Math.round(performance.now()-start),
       trainingSuccessful:!out.cancelled,parameters:MecaV2.parameters(network.shape),
       environment:'Web Worker CPU',runtime:navigator.userAgent||'worker',epochs:1,batchSize:64});
    send('benchmarkProgress',{result:results.at(-1)});
   }
   send('benchmarked',{results,cancelled:cancelRequested});
  }else throw Error('Ação desconhecida.');
 }catch(error){send('error',{action:req.action,message:String(error.message||error)});}
 finally{busy=false;}
};