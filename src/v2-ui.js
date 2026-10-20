/* MECA V2 local neural workbench. All controls operate on actual V2 state.
 * Existing Edge Impulse and legacy browser lab are intentionally separate.
 */
'use strict';
(()=>{
const V=window.MecaV2,root=document.querySelector('#neuralV2');
if(!root||!V)return;
const $=s=>root.querySelector(s);
const status=(message,bad=false)=>{$('#v2Status').textContent=message;$('#v2Status').className='v2Status'+(bad?' bad':'');};
const workerSupported=!!window.Worker;
let worker=null,model=null,dataReady=false,busy=false,counters={},history=[],lastSample=null,net=null;
let datasetInfo=null,trainingMs=null,lastEvaluation=null;
const cnv=$('#v2Canvas'),ctx=cnv.getContext('2d'),style={r:3,density:60,opacity:45,width:1,minMagnitude:0,scheme:'orange-blue',showLabels:true};
let shown=[],chosen=null,zoom=1,panX=0,panY=0,drag=null,drawing=null,imported=null;
const sync=()=>{for(const id of ['v2Generate','v2Train','v2Init','v2Benchmark','v2Save','v2LoadBtn','v2Test','v2Sample','v2Compare']){
 const el=$('#'+id);if(el)el.disabled=busy||(id==='v2Train'&&(!dataReady||!model))||(id==='v2Test'&&(!dataReady||!model))||(id==='v2Sample'&&(!dataReady||!model))||(id==='v2Compare'&&!dataReady)||(id==='v2Save'&&!model);
}$('#v2Cancel').disabled=!busy;};
function newWorker(){
 if(!workerSupported){status('Web Worker indisponível neste navegador; não há processamento simulado.',true);return;}
 worker?.terminate();
 try{worker=new Worker('src/v2-worker.js');worker.addEventListener('message',receive);
 worker.addEventListener('error',e=>{busy=false;sync();status('Erro no Worker: '+e.message,true);});}
 catch(e){status('Worker requer HTTP local. Execute python -m http.server 8000 e abra http://localhost:8000.',true);}
}
function parseShape(){
 const raw=$('#v2Hidden').value.trim().split(/[\s,;]+/).map(Number);
 if(raw.some(n=>!Number.isInteger(n)))throw Error('Camadas devem conter números inteiros.');
 return V.config({hidden:raw,activation:$('#v2Activation').value});
}
function opts(){
 return V.trainingOptions({epochs:Number($('#v2Epochs').value),lr:Number($('#v2Lr').value),batchSize:Number($('#v2Batch').value),
  optimizer:$('#v2Optimizer').value,seed:Number($('#v2Seed').value),patience:Number($('#v2Patience').value)});
}
function send(action,more={}){
 if(!worker)throw Error('Web Worker indisponível. Use servidor HTTP local.');
 worker.postMessage({action,...more});
}
function setBusy(v){busy=v;sync();}
function run(action,more){if(busy)return;try{setBusy(true);send(action,more);}catch(e){setBusy(false);status(e.message,true);}}
function fmt(n){return Number(n).toLocaleString('pt-BR',{maximumFractionDigits:3});}
function tms(n){return (n/1000).toFixed(2)+' s';}
function download(obj,name){const bytes=typeof obj==='string'?obj:JSON.stringify(obj,null,2);const blob=new Blob([bytes],{type:'application/json'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}
function receive({data:e}){
 try{
 switch(e.event){
 case 'error':setBusy(false);status(e.message,true);break;
 case 'generated':
  dataReady=true;setBusy(false);counters=e.counts;history=[];lastEvaluation=null;trainingMs=null;
  datasetInfo={...e.metadata,counts:e.counts,baselineOnValidation:e.baselineOnValidation,generationMs:e.generationMs,estimate:e.estimate};$('#v2Counts').textContent=Object.entries(counters).map(([k,v])=>k+': '+fmt(v)).join(' · ');
  $('#v2DatasetMeta').textContent='Seed '+e.metadata.seed+' · '+fmt(e.metadata.count)+' cenários sintéticos · gerados em '+tms(e.generationMs)+'.';
  status('Dados sintéticos gerados. Treinamento e teste continuam separados. Nenhuma coleta física foi executada.');
  break;
 case 'initialized':
  model=e.model;setBusy(false);history=model.trained?.history||[];lastSample=null;lastEvaluation=null;trainingMs=null;chosen=null;renderMetrics();render();
  status('Arquitetura inicializada: '+model.shape.join(' → ')+'. Pesos são matematicamente reais, mas ainda não comprovam aprendizado.');
  break;
 case 'epoch':
  history.push(e.stat);$('#v2Progress').value=e.stat.epoch/Number($('#v2Epochs').value)*100;
  $('#v2TrainMeta').textContent='Época '+e.stat.epoch+' · val. accuracy '+(e.stat.validation.accuracy*100).toFixed(1)+'% · loss '+e.stat.validation.loss.toFixed(4)+' · '+tms(e.elapsedMs);
  break;
 case 'trained':
  model=e.model;setBusy(false);history=e.history;lastSample=null;lastEvaluation=null;trainingMs=e.trainingMs;
  $('#v2Progress').value=100;renderMetrics();render();
  status('Treinamento concluído na CPU Web Worker em '+tms(e.trainingMs)+'. Métricas sintéticas não validam o robô físico.');
  break;
 case 'tested':
  setBusy(false);const d=e.result;lastEvaluation=d;
  $('#v2Results').textContent='Teste reservado: '+d.total+' amostras · accuracy '+(d.accuracy*100).toFixed(1)+'% · loss '+d.loss.toFixed(4)+
   '\nMatriz (real × previsto): '+JSON.stringify(d.matrix)+
   '\n'+d.classes.map(x=>x.label+' — precision '+x.precision.toFixed(3)+', recall '+x.recall.toFixed(3)+', F1 '+x.f1.toFixed(3)).join('\n');
  status('Resultado calculado sobre dados sintéticos reservados. Não reutilize o teste para escolher hiperparâmetros.');
  break;
 case 'sample':
  setBusy(false);lastSample=e.acts;$('#v2SampleInfo').textContent=e.sampleId+' · '+e.scenario+' · rótulo '+e.label+' · predição '+V.LABELS[e.acts.at(-1)[0]>=e.acts.at(-1)[1]?0:1];render();
  break;
 case 'benchmarkProgress':$('#v2Bench').textContent+='\\n'+JSON.stringify(e.result);break;
 case 'compared':
  setBusy(false);$('#v2Comparison').textContent=JSON.stringify({results:e.results,warning:e.warning,source:e.source},null,2);
  status('Comparação executada com o mesmo dataset reservado; modelos não alterados.');break;
 case 'benchmarked':setBusy(false);$('#v2Bench').textContent=JSON.stringify(e.results,null,2);status('Benchmarks CPU concluídos no Worker deste navegador.');break;
 case 'cancelled':setBusy(false);status('Treinamento cancelado após '+e.epochsCompleted+' épocas; modelo anterior preservado.');break;
 case 'cancelling':status('Cancelamento solicitado. Processamento será interrompido no próximo checkpoint.');break;
 }
 }catch(error){status(error.message,true);setBusy(false);}
}
function renderMetrics(){
 if(!model){$('#v2Architecture').textContent='Modelo não inicializado';return;}
 const shape=model.shape,count=V.parameters(shape);
 $('#v2Architecture').textContent='Arquitetura: '+shape.join(' → ')+' · '+fmt(count)+' parâmetros (pesos + bias) · '+model.activation;
 $('#v2ModelType').textContent=model.trained?'Treinado ('+model.trained.epochsCompleted+' épocas)':'Pesos inicializados';
}
function estimate(){
 try{const cfg=parseShape(),count=Number($('#v2Count').value),info=V.estimate(count,cfg.shape,Number($('#v2Epochs').value));
  $('#v2Estimate').textContent=fmt(info.parameters)+' parâmetros · ~'+info.estimatedDataMiB+' MiB dados brutos + '+info.estimatedFeatureMiB+' MiB features (estimativas mínimas) · '+fmt(info.estimatedOps)+' multiplicações-acumulações estimadas';
  $('#v2Generate').disabled=busy||count>10000||count<20;
 }catch(e){$('#v2Estimate').textContent=e.message;$('#v2Generate').disabled=true;}
}
['v2Hidden','v2Activation','v2Count','v2Epochs'].forEach(k=>$('#'+k).addEventListener('input',estimate));
$('#v2Preset').addEventListener('change',()=>{$('#v2Hidden').value=$('#v2Preset').value;estimate();});
$('#v2Init').addEventListener('click',()=>{try{const c=parseShape();
 if(model&&model.trained&&!confirm('Reinicializar? Pesos treinados não salvos serão perdidos.'))return;
 run('initialize',{options:{hidden:c.shape.slice(1,-1),activation:c.activation},seed:Number($('#v2Seed').value)});
 }catch(e){status(e.message,true);}});
$('#v2Generate').addEventListener('click',()=>{try{
 const c=parseShape(),count=Number($('#v2Count').value);
 if(model?.trained&&!confirm('Novo dataset será gerado; o modelo treinado anterior continuará no navegador, mas suas métricas não serão válidas para o novo conjunto. Continuar?'))return;
 run('generate',{options:{count,seed:Number($('#v2Seed').value),split:[
 Number($('#v2SplitTrain').value)/100,Number($('#v2SplitVal').value)/100,Number($('#v2SplitTest').value)/100],shape:c.shape,epochs:Number($('#v2Epochs').value)}});
 dataReady=false;
 }catch(e){status(e.message,true);setBusy(false);}});
$('#v2Train').addEventListener('click',()=>{try{if(model?.trained&&!confirm('Treinar a partir dos pesos atuais? O modelo salvo só muda após concluir.'))return;
 history=[];$('#v2Progress').value=0;run('train',{options:opts()});}catch(e){status(e.message,true);}});
$('#v2Cancel').addEventListener('click',()=>{if(busy)send('cancel');});
$('#v2Test').addEventListener('click',()=>run('test'));
$('#v2Sample').addEventListener('click',()=>run('sample',{index:Number($('#v2SampleIndex').value)}));
$('#v2Save').addEventListener('click',()=>{if(model)download(model,'meca-neural-v2-model.json');});
$('#v2Report').addEventListener('click',()=>{
 if(!model){status('Inicialize ou importe uma arquitetura primeiro.',true);return;}
 const id=$('#v2ExperimentId').value.trim();
 if(!/^[a-zA-Z0-9_.-]{1,80}$/.test(id)){status('Experimento: use 1 a 80 caracteres alfanuméricos, _ . ou -.',true);return;}
 const report={
  schema:'MIH-V2-EXPERIMENT-v1',experiment_id:id,
  created_at:new Date().toISOString(),origin:'synthetic_lab',
  architecture:{shape:model.shape,activation:model.activation,labels:model.labels,
    parameters:V.parameters(model.shape),model_origin:model.origin},
  dataset:datasetInfo,training:model.trained,
  environment:'browser-cpu-worker',trainingMs:trainingMs,
  evaluation:lastEvaluation,
  imported_file:imported?{origin:imported.origin,format:imported.format,compatible:imported.compatible,
    missing_channels:imported.missing,physical_origin_authenticated:false}:null,
  limits:['Resultados sintéticos não são validação no Mequinha físico.',
   'Origem física declarada em arquivos não é autenticada.',
   'Benchmark do runner GitHub não determina desempenho deste dispositivo.'],
  meca:{observation:'Resultado experimental computado na CPU local quando disponível.',
   hypothesis:'Diagnóstico neural físico permanece não verificado.',
   proposal:'Coletar múltiplas sessões reais, rotular e separar por execução antes de avaliar generalização.',
   decision:'Continuar coleta; não automatizar OpModes.'}
 };
 download(report,'meca-v2-experiment-'+id+'.json');
 status('Relatório exportado com dados realmente disponíveis; campos não executados permanecem null.');
});
$('#v2History').addEventListener('click',()=>{if(!history.length){status('Sem histórico de treinamento para exportar.',true);return;}
 download({source:'synthetic_lab',architecture:model.shape,history,environment:'browser-cpu-worker'},'meca-v2-training-history.json');});
$('#v2LoadBtn').addEventListener('click',()=>$('#v2ModelFile').click());
$('#v2ModelFile').addEventListener('change',async e=>{
 const f=e.target.files?.[0];if(!f)return;
 try{if(f.size>8e6)throw Error('Modelo maior que 8 MB; importação bloqueada.');
  const m=V.legacy(JSON.parse(await f.text()));if(model?.trained&&!confirm('Substituir o modelo atual pelo arquivo selecionado?'))return;
  $('#v2Hidden').value=m.shape.slice(1,-1).join(',');$('#v2Activation').value=m.activation;
  run('load',{model:m});estimate();
 }catch(err){status(err.message,true);}finally{e.target.value='';}
});
$('#v2ImportFile').addEventListener('change',async e=>{
 const file=e.target.files?.[0];if(!file)return;
 try{if(file.size>8*1024*1024)throw Error('Arquivo maior que 8 MiB.');
  imported=V.importTelemetry(await file.text(),file.name,$('#v2Origin').value);
  $('#v2ImportResult').textContent=JSON.stringify({file:file.name,origin:imported.origin,format:imported.format,
   samples:imported.rows,missing:imported.missing,compatible:imported.compatible,
   reason:imported.reason,preview:imported.preview},null,2);
  $('#v2InferFile').disabled=!(model&&imported.compatible);
  status(imported.reason,!imported.compatible);
 }catch(err){imported=null;$('#v2InferFile').disabled=true;status(err.message,true);}finally{e.target.value='';}
});
$('#v2InferFile').addEventListener('click',()=>{
 try{if(!model||!imported?.compatible)throw Error('Modelo/dados incompatíveis.');
  const acts=V.forward(model,imported.raw);lastSample=acts;
  $('#v2ImportPrediction').textContent=imported.origin.toUpperCase()+' · probabilidades: '+V.LABELS.map((s,i)=>s+': '+(acts.at(-1)[i]*100).toFixed(2)+'%').join(' · ')+
   ' · PREDIÇÃO EXPERIMENTAL, NÃO DIAGNÓSTICO CONFIRMADO.';
  render();
 }catch(err){status(err.message,true);}
});
$('#v2Compare').addEventListener('click',async()=>{
 try{
  const files=Array.from($('#v2CompareFiles').files||[]);
  if(files.length!==2)throw Error('Selecione exatamente dois arquivos de modelos.');
  const models=[];
  for(const f of files){
   if(f.size>8e6)throw Error('Arquivo de modelo excede 8 MB.');
   models.push(V.legacy(JSON.parse(await f.text())));
  }
  run('compare',{models});
 }catch(e){setBusy(false);status(e.message,true);}
});
$('#v2Benchmark').addEventListener('click',()=>{
 if(!confirm('Benchmark CPU de 1000, 5000 e 10000 amostras, 1 época por lote. Poderá consumir memória e CPU. Executar?'))return;
 $('#v2Bench').textContent='Executando benchmarks no Worker';run('benchmark');
});
function layerPositions(){
 if(!model)return [];
 const shape=model.shape,width=Math.max(640,cnv.clientWidth),height=Math.max(400,cnv.clientHeight);
 return shape.map((count,l)=>{
  const drawCount=Math.min(count,80),indices=Array.from({length:drawCount},(_,i)=>Math.floor((i+.5)*count/drawCount));
  return indices.map((idx,pos)=>({l,i:idx,x:55+l*(width-110)/(shape.length-1),
   y:35+(pos+.5)*(height-70)/drawCount}));
 });
}
function hit(x,y){
 let best=null,d=20;for(const layer of shown)for(const p of layer){
  const dist=Math.hypot(p.x-x,p.y-y);if(dist<d){best=p;d=dist;}
 }return best;
}
function details(){
 const el=$('#v2Neuron');
 if(!chosen||!model){el.textContent='Clique em um neurônio para inspecionar pesos, bias e ativação.';return;}
 const {l,i}=chosen,activation=lastSample?.[l]?.[i];
 const lines=['Camada '+l+' · neurônio '+(i+1),
  'Ativação: '+(activation===undefined?'sem amostra':activation.toFixed(6))];
 if(l>0){lines.push('Bias real: '+model.B[l-1][i].toFixed(6));
 const ordered=model.W[l-1][i].map((w,j)=>({source:j+1,w})).sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,12);
 lines.push(...ordered.map(p=>'Origem '+p.source+' → '+(i+1)+': '+p.w.toFixed(6)));}
 el.textContent=lines.join('\n');
}
function render(){
 if(!ctx||!model)return;
 const w=cnv.clientWidth,h=cnv.clientHeight,dpr=Math.min(2,window.devicePixelRatio||1);
 cnv.width=Math.round(w*dpr);cnv.height=Math.round(h*dpr);
 ctx.setTransform(dpr,0,0,dpr,panX*dpr,panY*dpr);ctx.scale(zoom,zoom);
 ctx.clearRect(-panX/zoom,-panY/zoom,w/zoom,h/zoom);ctx.fillStyle='#0b0e15';ctx.fillRect(-panX/zoom,-panY/zoom,w/zoom,h/zoom);
 shown=layerPositions();const all=V.parameters(model.shape)-model.shape.slice(1).reduce((a,b)=>a+b,0);
 const threshold=Math.max((100-style.density)/100,style.minMagnitude/100),opa=style.opacity/100;
 let displayed=0,positive=0,negative=0;
 const targets=shown.slice(1),totalCap=6500;
 for(let l=0;l<targets.length;l++){
  const before=shown[l],after=shown[l+1];const max=Math.max(...model.W[l].flat().map(Math.abs),1e-9);
  for(const b of after)for(const a of before){
   const weight=model.W[l][b.i][a.i],magnitude=Math.abs(weight)/max;
   const selected=chosen&&(chosen.l===l&&chosen.i===a.i||chosen.l===l+1&&chosen.i===b.i);
   if(chosen&&!selected)continue;
   // On a selected node always show its actual incident edges, regardless of density filter.
   if((!chosen&&magnitude<threshold)||displayed>=totalCap)continue;
   ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
   ctx.strokeStyle=weight<0?(style.scheme==='mono'?'rgba(154,164,179,':'rgba(139,171,218,')+Math.min(.85,opa*magnitude)+')':'rgba(255,151,65,'+Math.min(.9,opa*magnitude)+')';
   ctx.lineWidth=(selected?2.2:.35+.75*magnitude)*style.width;ctx.stroke();displayed++;if(weight<0)negative++;else positive++;
  }
 }
 let nodes=0;
 for(const layer of shown)for(const p of layer){
  const activation=lastSample?.[p.l]?.[p.i],strength=activation===undefined?.6:Math.min(1,.25+Math.abs(activation));
  ctx.beginPath();ctx.arc(p.x,p.y,chosen?.l===p.l&&chosen?.i===p.i?style.r+3:style.r,0,Math.PI*2);
  ctx.fillStyle=activation===undefined?'rgba(255,167,85,.7)':'rgba(255,167,85,'+strength+')';ctx.fill();nodes++;
 }
 ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillStyle='#e6b486';
 if(style.showLabels)shown.forEach((layer,l)=>ctx.fillText(l===0?'Entradas (200)':l===shown.length-1?'Saídas (2)':'Oculta '+l+' ('+model.shape[l]+')',layer[0].x,19));
 $('#v2GraphStats').textContent=nodes+'/'+model.shape.reduce((a,b)=>a+b,0)+' neurônios visíveis · '+displayed+'/'+all+' conexões visíveis (filtro real), '+positive+' positivas, '+negative+' negativas · limite de peso '+threshold.toFixed(2)+' · zoom '+zoom.toFixed(2)+'×';
 details();
}
cnv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(3,Math.max(.6,zoom*(e.deltaY>0?.9:1.1)));render();},{passive:false});
cnv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,moved:false,startX:panX,startY:panY};cnv.setPointerCapture(e.pointerId);});
cnv.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(Math.abs(dx)+Math.abs(dy)>5)drag.moved=true;
 if(drag.moved){panX=drag.startX+dx;panY=drag.startY+dy;render();}
});
cnv.addEventListener('pointerup',e=>{
 if(drag&&!drag.moved){const box=cnv.getBoundingClientRect();
 chosen=hit((e.clientX-box.left-panX)/zoom,(e.clientY-box.top-panY)/zoom);render();}
 drag=null;
});
$('#v2ResetView').addEventListener('click',()=>{zoom=1;panX=panY=0;chosen=null;render();});
$('#v2ResetStyle').addEventListener('click',()=>{
 $('#v2Radius').value='3';$('#v2Density').value='60';$('#v2Opacity').value='45';
 $('#v2LineWidth').value='1';$('#v2MinMagnitude').value='0';
 $('#v2ColorScheme').value='orange-blue';$('#v2ShowLabels').checked=true;
 try{localStorage.removeItem('meca-v2-visual-v1');}catch{}
 captureVisualPrefs();
});
function captureVisualPrefs(){
 style.r=Number($('#v2Radius').value);style.density=Number($('#v2Density').value);
 style.opacity=Number($('#v2Opacity').value);style.width=Number($('#v2LineWidth').value);
 style.minMagnitude=Number($('#v2MinMagnitude').value);
 style.scheme=$('#v2ColorScheme').value;style.showLabels=$('#v2ShowLabels').checked;
 try{localStorage.setItem('meca-v2-visual-v1',JSON.stringify(style));}catch{}
 render();
}
for(const id of ['v2Radius','v2Density','v2Opacity','v2LineWidth','v2MinMagnitude','v2ColorScheme','v2ShowLabels'])
 $('#'+id).addEventListener('input',captureVisualPrefs);
try{
 const preference=JSON.parse(localStorage.getItem('meca-v2-visual-v1')||'null');
 if(preference&&typeof preference==='object'){
  for(const [key,id] of [['r','v2Radius'],['density','v2Density'],['opacity','v2Opacity'],
      ['width','v2LineWidth'],['minMagnitude','v2MinMagnitude'],['scheme','v2ColorScheme']]){
   if(preference[key]!==undefined)
    $('#'+id).value=String(preference[key]);
  }
  $('#v2ShowLabels').checked=preference.showLabels!==false;
  captureVisualPrefs();
 }
}catch{}
const lessons=[
['1 — Anatomia da rede','As 200 entradas representam 40 leituras × 5 canais. Camadas ocultas aplicam transformações; duas saídas representam IMU travada e normal.','Abra uma arquitetura e clique em um neurônio.'],
['2 — Arquiteturas e custo','Mais camadas/pesos aumentam capacidade e custo; podem também levar ao sobreajuste.','Mude o preset e reinicialize. Observe a contagem de pesos.'],
['3 — Dados sintéticos','Treino atualiza pesos; validação escolhe configurações; teste reservado avalia generalização dentro do gerador.','Altere a quantidade de amostras e clique Gerar.'],
['4 — Otimização','Épocas percorrem os dados; batch controla a quantidade de exemplos por atualização; Adam e SGD fazem passos diferentes.','Configure épocas, learning rate, batch e inicie o treinamento.'],
['5 — Resultados','Accuracy sozinha não mostra falsos negativos. Use matriz de confusão, precision, recall, F1 e loss.','Avalie teste reservado após treinar e compare as classes.'],
['6 — Telemetria do Mequinha','JSON MIH-V1 e CSV precisam de 40 leituras a 20 Hz, encoders, IMU e potência. Campo ausente bloqueia a inferência.','Importe um arquivo real com origem declarada e examine a prévia.'],
['7 — Processamento','CPU Worker roda no dispositivo; GPU e VM remota não foram implementadas e não estão habilitadas.','Confira o ambiente de execução e execute um benchmark opcional.'],
['8 — Registro M.E.C.A.','Mapear → Explorar → Concretizar → Avaliar. Registre seed, arquitetura, versão, desempenho e limitações.','Exporte modelo e histórico. Nunca confunda simulação com validação física.']
];
const guide=$('#v2Guide');
for(const [title,body,action] of lessons){const d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('p'),a=document.createElement('p');
 s.textContent=title;p.textContent=body;a.textContent='Ação prática: '+action;d.append(s,p,a);guide.append(d);}
window.addEventListener('resize',()=>{if(model)render();});
$('#v2Environment').textContent=workerSupported?'CPU local · Web Worker (disponibilidade não comprova desempenho)':'Indisponível · requer Web Worker';
newWorker();estimate();sync();status('V2 experimental. Inicialize a arquitetura, gere os dados e treine. Dados não saem do navegador.');
})();