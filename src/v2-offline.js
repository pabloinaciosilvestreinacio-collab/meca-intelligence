/* Offline CPU fallback for file:// and browsers without Web Worker support.
 * Runs the same MecaV2 math; no mocked results, remote services or hidden training.
 * Limited to 5,000 synthetic samples to avoid blocking slower browsers.
 */
(function(root){
'use strict';
root.MecaOfflineRunner=function(){
 const V=root.MecaV2,core=root.MecaCore;
 if(!V||!core)throw Error('Motores matemáticos ausentes. Verifique todos os arquivos da pasta src.');
 let data=null,model=null,busy=false,cancel=false,closed=false;
 const listeners={message:[],error:[]};
 const emit=(event,payload={})=>{
  if(closed)return;
  const obj={data:{event,...payload}};
  for(const listener of listeners.message)listener(obj);
 };
 const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
 async function handle(req){
  try{
   if(req.action==='cancel'){
    cancel=true;emit('cancelling');return;
   }
   if(busy)throw Error('Existe uma tarefa em execução no modo offline.');
   if(req.action==='generate'){
    busy=true;const start=performance.now(),p=req.options;
    if(p.count>5000)throw Error('O modo de arquivo local suporta até 5.000 amostras. Use servidor HTTP/Pages para 10.000.');
    data=V.samples(p.count,p.seed,p.split);
    emit('generated',{counts:Object.fromEntries(['training','validation','testing'].map(k=>[k,data[k].length])),
     metadata:data.metadata,baselineOnValidation:core.evaluateBaseline(data.validation),
     estimate:V.estimate(p.count,p.shape,p.epochs),generationMs:Math.round(performance.now()-start)});
   }else if(req.action==='initialize'){
    model=V.init(req.options,req.seed);emit('initialized',{model});
   }else if(req.action==='load'){
    model=V.legacy(req.model);emit('initialized',{model});
   }else if(req.action==='trainConfigured'){
    // No Web Worker in file:// mode. Same numerical training, no fake outcomes.
    busy=true;cancel=false;
    const p=req.config,opts=V.trainingOptions(req.options);
    const networkConfig=V.config({hidden:p.hidden,activation:p.activation});
    if(p.count>5000||p.count<20)throw Error('Modo offline: selecione de 20 a 5.000 amostras; para 10.000 use servidor HTTP.');
    const started=performance.now();
    emit('preparing',{message:'Gerando '+p.count+' amostras e inicializando '+networkConfig.shape.join(' → ')+' na CPU offline.'});
    const pendingData=V.samples(p.count,p.seed,p.split);
    const candidate=V.init({hidden:p.hidden,activation:p.activation},p.seed);
    const generationMs=Math.round(performance.now()-started);
    if(cancel)emit('cancelled',{epochsCompleted:0});
    else{
     const trainingStarted=performance.now();
     const trained=await V.train(candidate,pendingData,{...opts,
      onEpoch:stat=>emit('epoch',{stat,elapsedMs:Math.round(performance.now()-trainingStarted)}),
      shouldStop:()=>cancel,yieldTick:tick});
     if(trained.cancelled)emit('cancelled',{epochsCompleted:trained.epochsCompleted});
     else{
      data=pendingData;model=trained.model;
      emit('trained',{model,history:trained.history,trainingMs:Math.round(performance.now()-trainingStarted),
       environment:'browser-cpu-offline-fallback',generationMs,
       counts:Object.fromEntries(['training','validation','testing'].map(k=>[k,data[k].length])),
       metadata:data.metadata,baselineOnValidation:core.evaluateBaseline(data.validation)});
     }
    }
   }else if(req.action==='train'){
    if(!data||!model)throw Error('Gere dados e inicialize a arquitetura antes do treino.');
    cancel=false;busy=true;
    const start=performance.now();
    const result=await V.train(model,data,{...req.options,
     onEpoch:stat=>emit('epoch',{stat,elapsedMs:Math.round(performance.now()-start)}),
     shouldStop:()=>cancel,yieldTick:tick});
    if(result.cancelled)emit('cancelled',{epochsCompleted:result.epochsCompleted});
    else{model=result.model;emit('trained',{model,history:result.history,
     trainingMs:Math.round(performance.now()-start),environment:'browser-cpu-offline-fallback'});}
   }else if(req.action==='test'){
    if(!data||!model)throw Error('Gere dados e inicialize o modelo.');
    emit('tested',{result:V.evaluate(model,data.testing),total:data.testing.length});
   }else if(req.action==='sample'){
    if(!data||!model)throw Error('Gere dados e inicialize o modelo.');
    const index=Math.max(0,Math.min(data.testing.length-1,Math.floor(req.index||0)));
    const sample=data.testing[index];
    emit('sample',{index,label:sample.label,scenario:sample.scenario,sampleId:sample.sampleId,
     acts:V.forward(model,V.sampleRaw(sample))});
   }else if(req.action==='compare'){
    if(!data)throw Error('Gere dados para comparar modelos.');
    if(!Array.isArray(req.models)||req.models.length!==2)throw Error('Selecione dois modelos.');
    busy=true;
    const results=req.models.map((incoming,index)=>{
     const m=V.legacy(incoming);
     return {index:index+1,shape:m.shape,parameters:V.parameters(m.shape),
      origin:m.origin,trained:m.trained!==null,metrics:V.evaluate(m,data.testing)};
    });
    emit('compared',{results,source:data.metadata,
     warning:'Resultados sintéticos, não validação física.'});
   }else if(req.action==='benchmark'){
    throw Error('Benchmark de 10.000 amostras requer o modo Worker HTTP. Abra o laboratório pelo endereço publicado.');
   }else throw Error('Ação desconhecida: '+req.action);
  }catch(e){emit('error',{action:req.action,message:String(e.message||e)});}
  finally{if(req.action!=='cancel')busy=false;}
 }
 return {
  addEventListener(type,fn){if(listeners[type])listeners[type].push(fn);},
  postMessage(req){if(req.action==='cancel'){cancel=true;emit('cancelling');return;}
   setTimeout(()=>{void handle(req);},0);},
  terminate(){closed=true;cancel=true;}
 };
};
})(window);