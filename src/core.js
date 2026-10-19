/* Meca Intelligence Browser Lab — synthetic learning experiments, not hardware diagnosis.
 * No external dependencies or telemetry uploads.
 */
(function(root,factory){'use strict';const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.MecaCore=api;})(typeof window!=='undefined'?window:globalThis,function(){
'use strict';
const LABELS=['imu_travada','normal'];
const HEADERS=['timestamp','imu_yaw_deg','encoder_md_ticks','encoder_me_ticks','potencia_md','potencia_me'];
const IN=200, H1=20,H2=10, OUT=2;
function rng(seed){let a=seed>>>0;return function(){a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};}
function gauss(r){return Math.sqrt(-2*Math.log(Math.max(1e-9,r())))*Math.cos(2*Math.PI*r());}
function pick(r,a){return a[Math.floor(r()*a.length)];}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}
/** 2 seconds at 20 Hz. Both labels require a known ground truth from the generator.
 * 'imu_travada' only occurs when rotation would make it observable. */
function synth(seed,label,scenario){
 const r=rng(seed),g=()=>gauss(r),fault=label===LABELS[0];
 const cases=fault?['giro','curva','lento','reversao','giro_ruidoso','intermitente']:
                   ['giro','curva','lento','reversao','giro_ruidoso','reta','parado','reta_assimetrica'];
 scenario=scenario||pick(r,cases);
 const sign=r()>.5?1:-1,mag=scenario==='lento'?.16+r()*.18:.38+r()*.48;
 const off=(r()-.5)*.24,noiseY=.025+r()*.30,noiseE=.25+r()*2.1;
 let yaw=(r()-.5)*95,left=0,right=0,stuckYaw=yaw,lastTrue=yaw;
 const rows=[];let startFreeze=fault?(scenario==='intermitente'?11:scenario==='reversao'?9:0):999;
 let endFreeze=fault?(scenario==='intermitente'?31:40):0;
 let mode='';
 for(let i=0;i<40;i++){
   let L=0,R=0, modulation=1+.17*Math.sin(i*.17+seed);
   if(scenario==='parado'){L=0;R=0;}
   else if(scenario==='reta'||scenario==='reta_assimetrica'){
    L=(.22+r()*.55)*sign;R=L+(scenario==='reta_assimetrica'?.024+r()*.036:(r()-.5)*.012);
   }else if(scenario==='reversao'){
     const direction=i<19?sign:-sign;L=direction*mag*.7;R=-direction*mag*.7;
   }else if(scenario==='curva'){
     L=sign*(.4-mag*.48); R=sign*(.4+mag*.48);
   }else{
     L=-sign*mag*.8;R=sign*mag*.8;
   }
   L=clamp(L*modulation+off,-1,1);R=clamp(R*modulation-off,-1,1);
   // 2WD kinematics: signed encoder ticks and yaw remain mutually consistent.
   let actualL=L*(.93+r()*.13),actualR=R*(.93+r()*.13);
   let slip=(scenario==='giro_ruidoso'? .7+r()*.45: .92+r()*.12);
   left+=actualL*650*.05 +g()*noiseE*.25; right+=actualR*650*.05+g()*noiseE*.25;
   lastTrue=yaw;
   yaw+=(actualR-actualL)*55*.05*slip;
   let measured=yaw+g()*noiseY;
   if(fault){
     if(i===startFreeze)stuckYaw=lastTrue;
     if(i>=startFreeze&&i<endFreeze)measured=stuckYaw+g()*noiseY*.14;
   }
   rows.push([i*50,+measured.toFixed(5),+(right+g()*noiseE).toFixed(4),+(left+g()*noiseE).toFixed(4),+R.toFixed(4),+L.toFixed(4)]);
 }
 return {seed,label,scenario,rows,origin:'synthetic',sampleId:`${label}.${String(seed).padStart(6,'0')}`};
}
function createDataset(seed=32578,config={training:600,validation:150,testing:150}){
 const counts={training:config.training??600,validation:config.validation??150,testing:config.testing??150};
 const out={training:[],validation:[],testing:[],seed,synthetic:true};let n=0;
 for(const split of ['training','validation','testing']){
   for(let i=0;i<counts[split];i++){
     const label=LABELS[i%2];const sampleSeed=(seed*97+1013904223+n*1013)>>>0;
     out[split].push(synth(sampleSeed,label));n++;
   }
 }
 return out;
}
function flatten(sample){return sample.rows.flatMap(row=>row.slice(1));}
function toFeatures(raw){
 if(raw.length!==IN)throw Error('A amostra precisa conter exatamente 40 leituras × 5 sensores.');
 const x=new Float32Array(IN);
 for(let t=0;t<40;t++){
  const k=t*5,kp=(t-1)*5;
  // Model trained here uses temporal increments (not the Edge Impulse model's raw inputs).
  x[k]=clamp((t?raw[k]-raw[kp]:0)/4,-4,4);
  x[k+1]=clamp((t?raw[k+1]-raw[kp+1]:0)/30,-4,4);
  x[k+2]=clamp((t?raw[k+2]-raw[kp+2]:0)/30,-4,4);
  x[k+3]=clamp(raw[k+3],-1,1);
  x[k+4]=clamp(raw[k+4],-1,1);
 }
 return x;
}
function infer(model,raw){
 const x=toFeatures(raw);let prev=x;const acts=[Array.from(x)];
 for(let l=0;l<3;l++){
  const next=model.W[l].map((row,j)=>{
   let value=model.B[l][j];for(let i=0;i<prev.length;i++)value+=row[i]*prev[i];
   return l<2?Math.max(0,value):value;
  });prev=next;acts.push(next);
 }
 const logits=acts[3],m=Math.max(...logits),exp=logits.map(v=>Math.exp(v-m)),s=exp[0]+exp[1];acts[3]=exp.map(v=>v/s);
 return acts;
}
function initModel(seed){const r=rng(seed);let sizes=[IN,H1,H2,OUT];return {kind:'browser',version:1,source:'Treinado localmente (sintético)',W:[0,1,2].map(l=>Array.from({length:sizes[l+1]},()=>Array.from({length:sizes[l]},()=>gauss(r)*Math.sqrt(2/sizes[l])))),B:[[...new Array(H1).fill(0)],[...new Array(H2).fill(0)],[0,0]],shape:sizes,labels:LABELS};}
function evalModel(model,samples){const cm=[[0,0],[0,0]];let loss=0;
 for(const s of samples){let p=infer(model,flatten(s))[3],target=LABELS.indexOf(s.label);if(target<0)continue;const pred=p[0]>=p[1]?0:1;cm[target][pred]++;loss-=Math.log(Math.max(p[target],1e-9));}
 const n=cm.flat().reduce((a,b)=>a+b,0),correct=cm[0][0]+cm[1][1];
 return {accuracy:n?correct/n:0,correct,total:n,loss:loss/Math.max(1,n),matrix:cm,perClass:LABELS.map((name,i)=>({name,recall:cm[i][i]/Math.max(1,cm[i][0]+cm[i][1])}))};
}
async function train({dataset,epochs=24,lr=.003,seed=32578,onEpoch=()=>{},yieldTick=()=>new Promise(ok=>setTimeout(ok,0)),model:inputModel=null}){
 if(!dataset?.training?.length||!dataset?.validation?.length)throw Error('Gere as amostras antes de treinar.');
 epochs=clamp(Math.floor(epochs),1,120);
 const model=inputModel||initModel(seed),sizes=model.shape||[IN,H1,H2,OUT],r=rng(seed^0xABC1);
 const layers=model.W.map((rows,l)=>({n:rows.length,m:rows[0].length,w:Float32Array.from(rows.flat()),b:Float32Array.from(model.B[l])}));
 const adam=layers.map(l=>({mw:new Float32Array(l.w.length),vw:new Float32Array(l.w.length),mb:new Float32Array(l.b.length),vb:new Float32Array(l.b.length)}));
 const samples=dataset.training.map(s=>({x:toFeatures(flatten(s)),y:LABELS.indexOf(s.label)}));
 if(samples.some(s=>s.y<0))throw Error('Rótulos não suportados.');
 let step=0;
 for(let epoch=1;epoch<=epochs;epoch++){
  for(let i=samples.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[samples[i],samples[j]]=[samples[j],samples[i]];}
  const batchSize=24;
  for(let off=0;off<samples.length;off+=batchSize){
   const gradients=layers.map(l=>({w:new Float32Array(l.w.length),b:new Float32Array(l.b.length)}));const n=Math.min(batchSize,samples.length-off);
   for(let si=off;si<off+n;si++){
    const sample=samples[si],act=[sample.x];
    for(let l=0;l<3;l++){
      const L=layers[l],prev=act[l],a=new Float32Array(L.n);
      for(let j=0;j<L.n;j++){
        let sum=L.b[j],base=j*L.m;
        for(let i=0;i<L.m;i++)sum+=L.w[base+i]*prev[i];
        a[j]=l<2?Math.max(0,sum):sum;
      } act.push(a);
    }
    const logits=act[3],mx=Math.max(logits[0],logits[1]),e0=Math.exp(logits[0]-mx),e1=Math.exp(logits[1]-mx),sum=e0+e1;
    let delta=new Float32Array([e0/sum-(sample.y===0?1:0),e1/sum-(sample.y===1?1:0)]);
    for(let l=2;l>=0;l--){
      const L=layers[l],G=gradients[l],prev=act[l];const prevDelta=new Float32Array(L.m);
      for(let j=0;j<L.n;j++){
        const d=delta[j];G.b[j]+=d;const base=j*L.m;
        for(let i=0;i<L.m;i++){G.w[base+i]+=d*prev[i];prevDelta[i]+=L.w[base+i]*d;}
      }
      if(l>0){for(let i=0;i<prevDelta.length;i++)if(act[l][i]<=0)prevDelta[i]=0;}
      delta=prevDelta;
    }
   }
   step++;
   for(let l=0;l<3;l++){
    const L=layers[l],G=gradients[l],A=adam[l];
    const apply=(params,grads,m,v)=>{for(let i=0;i<params.length;i++){
      const g=clamp(grads[i]/n,-3,3);m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g;
      params[i]-=lr*(m[i]/(1-Math.pow(.9,step)))/(Math.sqrt(v[i]/(1-Math.pow(.999,step)))+1e-8);
    }};
    apply(L.w,G.w,A.mw,A.vw);apply(L.b,G.b,A.mb,A.vb);
   }
  }
  for(let l=0;l<3;l++){
   const L=layers[l];model.W[l]=Array.from({length:L.n},(_,j)=>Array.from(L.w.subarray(j*L.m,(j+1)*L.m)));model.B[l]=Array.from(L.b);
  }
  const stat=evalModel(model,dataset.validation);onEpoch({epoch,epochs,...stat});await yieldTick();
 }
 model.trained={epochs,seed,trainingSamples:dataset.training.length,validationSamples:dataset.validation.length,source:'Synthetic browser training (not Edge Impulse)'};
 return model;
}
function csvForSample(sample){return HEADERS.join(',')+'\n'+sample.rows.map(row=>row.join(',')).join('\n')+'\n';}
function csvParse(text,label='normal',name='uploaded'){
 const ls=text.replace(/^\ufeff/,'').trim().split(/\r?\n/);const h=ls.shift().split(',').map(x=>x.trim());const idx=HEADERS.map(v=>h.indexOf(v));if(idx.some(x=>x<0))throw Error('Cabeçalho CSV inválido.');
 if(ls.length!==40)throw Error('Esperadas exatamente 40 linhas de leituras.');
 const rows=ls.map((ln,k)=>{const parts=ln.split(',');const row=idx.map(i=>Number(parts[i]));if(row.some(x=>!Number.isFinite(x)))throw Error(`CSV inválido na linha ${k+2}`);return row;});
 if(rows.some((row,i)=>Math.abs(row[0]-i*50)>1))throw Error('Tempo deve aumentar de 50 em 50 ms (20 Hz).');
 return {seed:null,label,scenario:'importado',rows,origin:'uploaded',sampleId:name};
}
function baseline(s){const a=s.rows,z=a.length-1,dYaw=Math.abs(a[z][1]-a[0][1]),diff=Math.abs((a[z][2]-a[0][2])-(a[z][3]-a[0][3]));return diff>65&&dYaw<1?'imu_travada':'normal';}
function evaluateBaseline(samples){const cm=[[0,0],[0,0]];for(const s of samples){const i=LABELS.indexOf(s.label),j=LABELS.indexOf(baseline(s));if(i>=0)cm[i][j]++;}return {matrix:cm,accuracy:(cm[0][0]+cm[1][1])/samples.length};}
/** Builds ZIP (STORE method, CRC32), compatible with browsers and Edge Impulse CSV uploads. */
function zipFiles(files){const encoder=new TextEncoder(),chunks=[],central=[];let cursor=0;const table=new Uint32Array(256);
 for(let n=0;n<256;n++){let c=n;for(let j=0;j<8;j++)c=c&1?(0xEDB88320^(c>>>1)):(c>>>1);table[n]=c>>>0;}
 function crc(bytes){let c=0xFFFFFFFF;for(const b of bytes)c=table[(c^b)&255]^(c>>>8);return(c^0xFFFFFFFF)>>>0;}
 function u16(x){return[x&255,(x>>>8)&255];} function u32(x){return[x&255,(x>>>8)&255,(x>>>16)&255,(x>>>24)&255];}
 for(const [filename,content] of files){const name=encoder.encode(filename),body=encoder.encode(content),check=crc(body),offset=cursor;
  const hdr=Uint8Array.from([80,75,3,4,...u16(20),...u16(0),...u16(0),...u16(0),...u16(0),...u32(check),...u32(body.length),...u32(body.length),...u16(name.length),...u16(0)]);
  chunks.push(hdr,name,body);cursor+=hdr.length+name.length+body.length;
  const cd=Uint8Array.from([80,75,1,2,...u16(20),...u16(20),...u16(0),...u16(0),...u16(0),...u16(0),...u32(check),...u32(body.length),...u32(body.length),...u16(name.length),...u16(0),...u16(0),...u16(0),...u16(0),...u32(0),...u32(offset)]);
  central.push(cd,name);
 }
 const cLen=central.reduce((a,v)=>a+v.length,0),count=files.length;
 const end=Uint8Array.from([80,75,5,6,...u16(0),...u16(0),...u16(count),...u16(count),...u32(cLen),...u32(cursor),...u16(0)]);
 return new Blob([...chunks,...central,end],{type:'application/zip'});
}
function datasetZip(data){const files=[];
 for(const s of ['training','validation','testing'])for(const x of data[s]){const folder=s==='testing'?'testing':'training';
  files.push([`${folder}/${x.sampleId}.csv`,csvForSample(x)]);
 }
 files.push(['MANIFESTO.txt',`MECA INTELLIGENCE: ${data.training.length} treino, ${data.validation.length} validação, ${data.testing.length} teste.\n`+
 'Todos são dados sintéticos. O ZIP inclui arquivos para Edge Impulse (pastas training/testing).\n'+
 'Treinamento+validação são combinados na pasta training ao importar no Edge Impulse.\n'+
 'Não use estas amostras como evidência de validação em robô real.\n']);
 return zipFiles(files);
}
return {LABELS,HEADERS,synth,createDataset,flatten,toFeatures,infer,train,initModel,evalModel,csvForSample,csvParse,baseline,evaluateBaseline,zipFiles,datasetZip};
});