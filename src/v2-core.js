/* MECA Neural Lab V2 — configurable dense networks, CPU reference engine.
 * No remote upload, no actuators, no claims of physical validation.
 */
(function(root,factory){'use strict';const api=factory(root);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.MecaV2=api;
})(typeof self!=='undefined'?self:globalThis,function(root){
'use strict';
const LABELS=['imu_travada','normal'],INPUT=200,MAX_PARAMS=100000;
function random(seed=32578){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-10,r())))*Math.cos(2*Math.PI*r());}
function parameters(shape){return shape.slice(1).reduce((sum,n,i)=>sum+n*(shape[i]+1),0);}
function config(input={}){
 const hidden=input.hidden??[20,10],activation=input.activation??'relu';
 if(!Array.isArray(hidden)||hidden.length<1||hidden.length>4||hidden.some(n=>!Number.isInteger(n)||n<2||n>128))throw Error('Use 1 a 4 camadas ocultas com 2 a 128 neurônios cada.');
 if(!['relu','tanh'].includes(activation))throw Error('Ativação inválida: use relu ou tanh.');
 const shape=[INPUT,...hidden,2];
 if(parameters(shape)>MAX_PARAMS)throw Error('Arquitetura excede 100.000 parâmetros; reduza as camadas.');
 return {shape,activation,labels:LABELS.slice(),parameters:parameters(shape),inputFormat:'40x5-delta-v1'};
}
function validateModel(m){
 if(!m||m.kind!=='browser'&&! (m.kind==='meca-v2'))throw Error('Formato de modelo incompatível.');
 if(!Array.isArray(m.shape)||m.shape[0]!==INPUT||m.shape.at(-1)!==2)throw Error('Entradas e classes incompatíveis com os sensores/labels.');
 const cfg=config({hidden:m.shape.slice(1,-1),activation:m.activation||'relu'});
 if(JSON.stringify(cfg.shape)!==JSON.stringify(m.shape))throw Error('Dimensões inválidas.');
 if(!Array.isArray(m.W)||!Array.isArray(m.B)||m.W.length!==m.shape.length-1||m.B.length!==m.W.length)throw Error('Camadas de pesos incompatíveis.');
 for(let l=0;l<m.W.length;l++){
  if(!Array.isArray(m.W[l])||m.W[l].length!==m.shape[l+1]||!Array.isArray(m.B[l])||m.B[l].length!==m.shape[l+1])throw Error('Número de neurônios incompatível.');
  for(const row of m.W[l])if(!Array.isArray(row)||row.length!==m.shape[l]||!row.every(Number.isFinite))throw Error('Pesos ou dimensões inválidos.');
  if(!m.B[l].every(Number.isFinite))throw Error('Bias inválido.');
 }
 if(JSON.stringify(m.labels||LABELS)!==JSON.stringify(LABELS))throw Error('Classes incompatíveis.');
 return true;
}
function init(input={},seed=32578){
 const c=config(input),r=random(seed),W=[],B=[];
 for(let l=0;l<c.shape.length-1;l++){
  const fan=c.shape[l],count=c.shape[l+1],scale=Math.sqrt((c.activation==='relu'?2:1)/fan);
  W.push(Array.from({length:count},()=>Array.from({length:fan},()=>normal(r)*scale)));
  B.push(Array(count).fill(0));
 }
 return {kind:'meca-v2',version:2,shape:c.shape,labels:c.labels,activation:c.activation,
  inputFormat:c.inputFormat,initSeed:seed,W,B,trained:null,origin:'initialized_untrained'};
}
function legacy(m){
 validateModel(m);
 return {kind:'meca-v2',version:2,shape:m.shape.slice(),labels:LABELS.slice(),
  activation:m.activation||'relu',inputFormat:'40x5-delta-v1',initSeed:m.initSeed??null,
  W:m.W.map(layer=>layer.map(r=>r.slice())),B:m.B.map(x=>x.slice()),
  trained:m.trained??null,origin:m.kind==='browser'?'legacy_browser_import':'v2_import'};
}
function feature(raw){
 const x=Array.isArray(raw)||ArrayBuffer.isView(raw)?Array.from(raw):[];
 if(x.length!==INPUT||!x.every(Number.isFinite))throw Error('Esperadas 40 leituras × 5 canais numéricos válidos.');
 const out=new Float64Array(INPUT);
 for(let t=0;t<40;t++){const k=t*5,p=k-5;
  out[k]=Math.max(-4,Math.min(4,(t?x[k]-x[p]:0)/4));
  out[k+1]=Math.max(-4,Math.min(4,(t?x[k+1]-x[p+1]:0)/30));
  out[k+2]=Math.max(-4,Math.min(4,(t?x[k+2]-x[p+2]:0)/30));
  out[k+3]=Math.max(-1,Math.min(1,x[k+3]));
  out[k+4]=Math.max(-1,Math.min(1,x[k+4]));
 }
 return out;
}
function forwardFeature(m,x,withCache=true){
 if(x.length!==m.shape[0])throw Error('Entrada diferente da arquitetura.');
 let a=Array.from(x),acts=[a],zs=[];
 for(let l=0;l<m.W.length;l++){
  const z=m.W[l].map((weights,j)=>{let v=m.B[l][j];for(let i=0;i<weights.length;i++)v+=weights[i]*a[i];return v;});
  if(!z.every(Number.isFinite))throw Error('Ativação não finita; ajuste parâmetros.');
  zs.push(z);
  if(l===m.W.length-1){
   const max=Math.max(...z),p=z.map(v=>Math.exp(v-max)),sum=p.reduce((v,t)=>v+t,0);
   a=p.map(v=>v/sum);
  }else a=z.map(v=>m.activation==='tanh'?Math.tanh(v):Math.max(0,v));
  acts.push(a);
 }
 return withCache?{acts,zs}:acts.at(-1);
}
function forward(m,raw){validateModel(m);return forwardFeature(m,feature(raw)).acts;}
function gradients(m,x,target){
 const {acts}=forwardFeature(m,x),out=acts.at(-1);
 if(target!==0&&target!==1)throw Error('Classe desconhecida.');
 const loss=-Math.log(Math.max(1e-12,out[target])),gw=new Array(m.W.length),gb=new Array(m.W.length);
 let delta=out.map((v,i)=>v-(target===i?1:0));
 for(let l=m.W.length-1;l>=0;l--){
  const w=m.W[l];gw[l]=w.map((row,j)=>row.map((_,i)=>delta[j]*acts[l][i]));
  gb[l]=delta.slice();
  if(l>0){
   const prev=Array(m.shape[l]).fill(0);
   for(let j=0;j<w.length;j++)for(let i=0;i<w[j].length;i++)prev[i]+=delta[j]*w[j][i];
   delta=prev.map((v,i)=>v*(m.activation==='tanh'?1-acts[l][i]**2:(acts[l][i]>0?1:0)));
  }
 }
 return {gw,gb,loss,probabilities:out};
}
function sampleRaw(s){if(!s||!Array.isArray(s.rows)||s.rows.length!==40)throw Error('Amostra precisa de 40 leituras.');return s.rows.flatMap(r=>r.slice(1));}
function samples(count,seed=32578,split=[.7,.15,.15]){
 if(!Number.isInteger(count)||count<20||count>25000)throw Error('Amostras entre 20 e 25.000.');
 if(!Number.isSafeInteger(seed))throw Error('Seed deve ser inteiro seguro.');
 if(!Array.isArray(split)||split.length!==3||split.some(v=>!Number.isFinite(v)||v<=0)||Math.abs(split.reduce((a,b)=>a+b,0)-1)>1e-8)throw Error('Split inválido.');
 const n1=Math.floor(count*split[0]),n2=Math.floor(count*split[1]),n3=count-n1-n2;
 if(Math.min(n1,n2,n3)<2)throw Error('Divisão precisa reservar ao menos duas amostras por conjunto.');
 const lab=root.MecaCore||(typeof require==='function'?require('./core.js'):null);
 if(!lab?.createDataset)throw Error('Gerador legado indisponível.');
 const ds=lab.createDataset(seed,{training:n1,validation:n2,testing:n3});
 ds.metadata={seed,count,split,source:'synthetic_lab',generator:'MecaCore.createDataset-v1',
  warning:'Todos os conjuntos vêm do mesmo gerador; não comprovam generalização física.'};
 return ds;
}
function estimate(count,shape=[200,20,10,2],epochs=6){
 const param=parameters(shape),rows=count*40;
 return {samples:count,parameters:param,estimatedDataMiB:Math.round(count*40*6*8/1048576*10)/10,
  estimatedFeatureMiB:Math.round(count*INPUT*8/1048576*10)/10,
  multiplyAddsPerEpoch:count*param,epochs,estimatedOps:count*param*epochs};
}
function evaluate(m,samples){
 const cm=[[0,0],[0,0]];let loss=0,valid=0;
 for(const s of samples){const t=LABELS.indexOf(s.label);if(t<0)continue;const p=forwardFeature(m,feature(sampleRaw(s)),false);
  const pred=p[0]>=p[1]?0:1;cm[t][pred]++;loss-=Math.log(Math.max(1e-12,p[t]));valid++;}
 const classes=[0,1].map(i=>{const tp=cm[i][i],fp=cm[1-i][i],fn=cm[i][1-i];
  const precision=tp/(tp+fp||1),recall=tp/(tp+fn||1);
  return {label:LABELS[i],precision,recall,f1:2*precision*recall/(precision+recall||1),support:tp+fn};});
 return {total:valid,accuracy:(cm[0][0]+cm[1][1])/(valid||1),
  loss:loss/(valid||1),matrix:cm,classes};
}
function trainingOptions(o={}){
 const epochs=Number(o.epochs??6),batchSize=Number(o.batchSize??24),lr=Number(o.lr??.003),
 seed=Number(o.seed??32578),patience=Number(o.patience??0),optimizer=o.optimizer??'adam';
 if(!Number.isInteger(epochs)||epochs<1||epochs>120)throw Error('Épocas entre 1 e 120.');
 if(!Number.isInteger(batchSize)||batchSize<1||batchSize>256)throw Error('Batch size entre 1 e 256.');
 if(!Number.isFinite(lr)||lr<=0||lr>.5)throw Error('Learning rate entre 0 e 0,5.');
 if(!Number.isSafeInteger(seed))throw Error('Seed inválida.');
 if(!Number.isInteger(patience)||patience<0||patience>40)throw Error('Patience entre 0 e 40.');
 if(!['adam','sgd'].includes(optimizer))throw Error('Otimizador não suportado.');
 return {epochs,batchSize,lr,seed,patience,optimizer};
}
async function train(m,dataset,options={}){
 validateModel(m);const o=trainingOptions(options);
 if(!dataset?.training?.length||!dataset?.validation?.length)throw Error('Dados insuficientes para treino/validação.');
 const onEpoch=options.onEpoch||(()=>{}),shouldStop=options.shouldStop||(()=>false);
 const yieldTick=options.yieldTick||(()=>new Promise(resolve=>setTimeout(resolve,0)));
 const cloned=legacy(m),trainData=dataset.training.map(s=>({x:feature(sampleRaw(s)),y:LABELS.indexOf(s.label)}));
 if(trainData.some(s=>s.y<0))throw Error('Rótulos de treino incompatíveis.');
 const r=random(o.seed),history=[],velocity=cloned.W.map(layer=>layer.map(row=>row.map(()=>0))),v2=cloned.W.map(layer=>layer.map(row=>row.map(()=>0))),mb=cloned.B.map(b=>b.map(()=>0)),vb=cloned.B.map(b=>b.map(()=>0));
 let step=0,best=Infinity,stale=0,bestModel=null;
 for(let epoch=1;epoch<=o.epochs;epoch++){
  if(shouldStop())return {cancelled:true,epochsCompleted:history.length,history};
  for(let i=trainData.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[trainData[i],trainData[j]]=[trainData[j],trainData[i]];}
  let loss=0;
  for(let offset=0;offset<trainData.length;offset+=o.batchSize){
   if(shouldStop())return {cancelled:true,epochsCompleted:history.length,history};
   const subset=trainData.slice(offset,offset+o.batchSize);
   const sumW=cloned.W.map(layer=>layer.map(row=>row.map(()=>0)));
   const sumB=cloned.B.map(row=>row.map(()=>0));
   for(const ex of subset){const g=gradients(cloned,ex.x,ex.y);loss+=g.loss;
    for(let l=0;l<sumW.length;l++)for(let j=0;j<sumW[l].length;j++){
     sumB[l][j]+=g.gb[l][j];
     for(let i=0;i<sumW[l][j].length;i++)sumW[l][j][i]+=g.gw[l][j][i];
    }
   }
   step++;
   const update=(param,g,l,j,i,isBias)=>{
    const delta=g/subset.length;
    if(o.optimizer==='sgd')return param-o.lr*delta;
    const first=isBias?mb[l]:velocity[l][j],second=isBias?vb[l]:v2[l][j],idx=isBias?j:i;
    first[idx]=.9*first[idx]+.1*delta;second[idx]=.999*second[idx]+.001*delta*delta;
    return param-o.lr*(first[idx]/(1-Math.pow(.9,step)))/(Math.sqrt(second[idx]/(1-Math.pow(.999,step)))+1e-8);
   };
   for(let l=0;l<cloned.W.length;l++)for(let j=0;j<cloned.W[l].length;j++){
    cloned.B[l][j]=update(cloned.B[l][j],sumB[l][j],l,j,0,true);
    for(let i=0;i<cloned.W[l][j].length;i++)cloned.W[l][j][i]=update(cloned.W[l][j][i],sumW[l][j][i],l,j,i,false);
   }
   if((Math.floor(offset/o.batchSize)%8)===0)await yieldTick();
  }
  const validation=evaluate(cloned,dataset.validation),stat={epoch,trainingLoss:loss/trainData.length,validation};
  if(!Number.isFinite(stat.trainingLoss)||!Number.isFinite(validation.loss))throw Error('Loss não finita, treino interrompido.');
  history.push(stat);onEpoch(stat);
  if(validation.loss<best-1e-6){best=validation.loss;stale=0;bestModel=JSON.parse(JSON.stringify({W:cloned.W,B:cloned.B}));}
  else stale++;
  await yieldTick();
  if(o.patience>0&&stale>=o.patience)break;
 }
 if(bestModel&&o.patience>0){cloned.W=bestModel.W;cloned.B=bestModel.B;}
 cloned.origin='trained_synthetic_lab';
 cloned.trained={options:o,epochsCompleted:history.length,history,source:dataset.metadata?.source||'unknown',generator:dataset.metadata?.generator||'unknown',
  validationBestLoss:best,warning:'Nenhum resultado representa validação no robô físico.'};
 validateModel(cloned);
 return {cancelled:false,model:cloned,history};
}
function importTelemetry(content,name='',origin){
 if(typeof content!=='string'||content.length>8*1024*1024)throw Error('Arquivo inválido ou maior que 8 MiB.');
 if(!['real','simulation','synthetic_lab'].includes(origin))throw Error('Escolha a origem declarada.');
 let rows=[],format,missing=[];
 const fields=['imu_yaw_deg','encoder_md_ticks','encoder_me_ticks','motor_md_command','motor_me_command'];
 if(name.toLowerCase().endsWith('.json')){
  let data;try{data=JSON.parse(content);}catch{throw Error('JSON inválido.');}
  if(data?.schema_version==='MIH-V1'&&Array.isArray(data.samples)){
   if(data.source!==origin)throw Error('A origem informada difere do arquivo.');
   const s=data.samples,normalized=[];
   for(let i=0;i<s.length;i++){
    const row=s[i];if(!row||typeof row!=='object')throw Error('Amostra JSON inválida.');
    const v=[row.time_ms,...fields.map(k=>row[k]??null)];
    normalized.push(v);
   }rows=normalized;format='MIH-V1';
  }else if(Array.isArray(data?.columns)&&Array.isArray(data?.rows)){
   const keys=data.columns,converted=data.rows.map(r=>Object.fromEntries(keys.map((k,i)=>[k,r[i]])));
   return importTelemetry([keys.join(','),...data.rows.map(r=>r.join(','))].join('\n'),'meca-sim.csv',origin);
  }else throw Error('JSON não é MIH-V1 ou recording do Meca Sim.');
 }else{
  const lines=content.replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(Boolean);
  const head=lines.shift().split(',').map(x=>x.trim());
  if(new Set(head).size!==head.length)throw Error('Colunas duplicadas no CSV.');
  const at=n=>head.indexOf(n);
  const sim=at('time_s')>=0&&at('yaw_deg')>=0;
  if(sim&&origin!=='simulation')throw Error('CSV de trajetória do Meca Sim exige origem simulação.');
  const lab=at('timestamp')>=0&&at('potencia_me')>=0;
  const time=sim?'time_s':lab?'timestamp':'time_ms';
  if(at(time)<0)throw Error('Coluna temporal ausente.');
  rows=lines.map((ln,i)=>{
   const columns=ln.split(',');if(columns.length!==head.length)throw Error('Linha '+(i+2)+' com colunas incorretas.');
   const get=k=>{const index=at(k);return index<0||columns[index].trim()===''?null:Number(columns[index]);};
   return [get(time)*(sim?1000:1),get(sim?'unsupported_imu': 'imu_yaw_deg'),
    get(sim?'MD_ticks':'encoder_md_ticks'),get(sim?'ME_ticks':'encoder_me_ticks'),
    get(lab?'potencia_md':'motor_md_command'),get(lab?'potencia_me':'motor_me_command')];
  });
  format=sim?'meca-sim-recording':lab?'browser-lab-csv':'mih-csv';
 }
 if(rows.length<1||rows.length>100000)throw Error('Limite: 1 a 100.000 linhas.');
 let previous=-Infinity;
 for(let i=0;i<rows.length;i++){
  const r=rows[i];if(!Array.isArray(r)||r.length!==6||r[0]===null||!Number.isFinite(r[0])||r[0]<0||r[0]<=previous)throw Error('Tempo inválido/não crescente na linha '+(i+1));
  previous=r[0];
  for(let j=1;j<6;j++)if(r[j]!==null&&!Number.isFinite(r[j]))throw Error('Valor inválido na linha '+(i+1));
  for(let j=4;j<6;j++)if(r[j]!==null&&Math.abs(r[j])>1)throw Error('Potência fora de [-1, 1].');
 }
 missing=fields.filter((k,i)=>rows.some(r=>r[i+1]===null));
 let compatible=rows.length===40&&missing.length===0;
 if(compatible)compatible=rows.every((r,i)=>Math.abs(r[0]-rows[0][0]-50*i)<=1);
 let reason=compatible?'Compatível com 40 × 5 leituras a 20 Hz.':
  missing.length?'Canais ausentes: '+missing.join(', ')+'; nenhuma medição foi inventada.':
  'Modelo espera exatamente 40 amostras a 20 Hz; reamostragem não implementada.';
 return {origin,format,rows:rows.length,missing,compatible,reason,
  preview:rows.slice(0,5),raw:compatible?rows.map(r=>r.slice(1)).flat():null,
  validation:'Formato verificado; origem declarada pelo operador, sem autenticação física.'};
}
return {LABELS,INPUT,MAX_PARAMS,config,parameters,init,legacy,validateModel,random,feature,
 forward,forwardFeature,gradients,sampleRaw,samples,estimate,evaluate,trainingOptions,train,importTelemetry};
});