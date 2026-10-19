'use strict';
// Executes entirely client-side. No model, measurements or training CSVs leave the browser.
const Lab=window.MecaCore;
let dataset=null,trainedModel=null,busy=false,currentSample=null;
const labEl=s=>document.querySelector(s);
const setLabStatus=(msg,kind='')=>{const el=labEl('#labStatus');el.textContent=msg;el.className='status'+(kind?' '+kind:'');};
function btnBusy(value){busy=value;for(const id of ['generateBtn','trainBtn','testBtn','exportDataBtn','exportModelBtn','loadWebModel','labSampleBtn']){const b=labEl('#'+id);if(b)b.disabled=value;}}
function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function updateNumbers(){if(!dataset)return;for(const k of ['training','validation','testing'])labEl('#count_'+k).textContent=dataset[k].length.toLocaleString('pt-BR');}
labEl('#generateBtn').addEventListener('click',()=>{
 try{dataset=Lab.createDataset(32578,{training:600,validation:150,testing:150});trainedModel=null;updateNumbers();
  const b=Lab.evaluateBaseline(dataset.testing);
  setLabStatus('900 cenários sintéticos gerados com divisões independentes. Agora clique em Treinar rede. A versão do Edge Impulse não será alterada.','ok');
  labEl('#labBaseline').textContent=`Regra simples no teste: ${(100*b.accuracy).toFixed(1)}% (referência, não IA)`;
  labEl('#labMetrics').textContent='Rede não treinada. Conjuntos de teste ainda não avaliados pela rede.';
  showSample(dataset.testing[0],false);
 }catch(err){setLabStatus(err.message,'bad');}
});
function showSample(sample,run=true){currentSample=sample;labEl('#labSampleInfo').textContent=`${sample.sampleId} · ${sample.scenario} · rótulo conhecido: ${sample.label} · sintético`;
 if(run&&model){try{inputValues=Lab.flatten(sample);activations=forward(inputValues);csvName=sample.sampleId+'.csv';updateOutputs();selected=null;updateDetail();draw();
 const p=activations[3],pred=p[0]>=p[1]?'imu_travada':'normal';labEl('#labSampleInfo').textContent+=` · previsão ${pred} (${(100*Math.max(...p)).toFixed(1)}%)`;
 }catch(err){setLabStatus('O modelo não avaliou este exemplo: '+err.message,'bad');}}
}
labEl('#trainBtn').addEventListener('click',async()=>{
 if(busy)return;if(!dataset){setLabStatus('Primeiro clique em Gerar 900 cenários.','bad');return;}
 btnBusy(true);
 try{
   const epochs=Number(labEl('#epochs').value);setLabStatus('Treinando MLP independente do Edge Impulse no navegador...');
   trainedModel=await Lab.train({dataset,epochs,lr:.003,onEpoch:o=>{
      labEl('#labProgress').value=o.epoch/o.epochs*100;
      labEl('#labMetrics').textContent=`Época ${o.epoch}/${o.epochs} · Validação: ${(o.accuracy*100).toFixed(1)}% · Loss: ${o.loss.toFixed(3)}`;
   }});
   model=trainedModel;selected=null;hover=null;inputValues=null;activations=null;csvName='';
   document.querySelector('#networkState').textContent='Modelo MLP treinado aqui no navegador — não é o modelo Edge Impulse';
   document.querySelector('#fmt').textContent='JS Float32 (treinado localmente)';
   document.querySelector('#modelStatus').textContent='MLP local';
   state('✓ Modelo MLP local treinado com dados sintéticos. Seus pesos são independentes do Edge Impulse. Exporte o modelo para guardá-lo.',true);
   setLabStatus('Treinamento concluído. Use Avaliar teste independente para medir a generalização.','ok');
   draw();
 }catch(err){setLabStatus(err.message,'bad');}
 finally{btnBusy(false);}
});
labEl('#testBtn').addEventListener('click',()=>{
 if(!dataset||!model){setLabStatus('Gere os cenários e carregue/treine um modelo antes do teste.','bad');return;}
 try{
  if(model.kind==='browser'){
   const x=Lab.evalModel(model,dataset.testing);
   labEl('#labTest').innerHTML=`<strong>${(100*x.accuracy).toFixed(2)}% · ${x.correct}/${x.total} acertos</strong> · Matriz [real × previsto]:<br>IMU travada: ${x.matrix[0][0]} como travada / ${x.matrix[0][1]} como normal<br>Normal: ${x.matrix[1][0]} como travada / ${x.matrix[1][1]} como normal`;
  }else{
   const cm=[[0,0],[0,0]];for(const s of dataset.testing){const a=forward(Lab.flatten(s))[3];const truth=Lab.LABELS.indexOf(s.label),pred=a[0]>=a[1]?0:1;cm[truth][pred]++;}
   const correct=cm[0][0]+cm[1][1];labEl('#labTest').innerHTML=`<strong>${(100*correct/dataset.testing.length).toFixed(2)}% · ${correct}/${dataset.testing.length}</strong> · Inferência INT8 aproximada do visualizador, não motor Edge Impulse.<br>IMU travada: ${cm[0].join(' / ')}; Normal: ${cm[1].join(' / ')}`;
  }
  setLabStatus('Teste independente executado. Esses são cenários sintéticos, sem validação no hardware real.','ok');
 }catch(err){setLabStatus(err.message,'bad');}
});
labEl('#labSampleBtn').addEventListener('click',()=>{
 if(!dataset){setLabStatus('Gere os cenários primeiro.','bad');return;}
 const arr=dataset.testing;const idx=Math.floor(Math.random()*arr.length);showSample(arr[idx]);
});
labEl('#exportDataBtn').addEventListener('click',()=>{
 if(!dataset){setLabStatus('Gere os cenários antes de exportar.','bad');return;}
 setLabStatus('Criando ZIP dos 900 cenários no próprio navegador...');
 setTimeout(()=>{try{const zip=Lab.datasetZip(dataset);download(zip,'meca-intelligence-900-cenarios-sinteticos.zip');setLabStatus('ZIP de 900 amostras criado. Nenhum dado enviado a servidores.','ok');}catch(err){setLabStatus(err.message,'bad');}},30);
});
labEl('#exportModelBtn').addEventListener('click',()=>{
 if(!trainedModel){setLabStatus('Primeiro treine uma MLP no navegador. O SDK Edge Impulse não pode ser redistribuído por aqui.','bad');return;}
 download(new Blob([JSON.stringify(trainedModel)],{type:'application/json'}),'meca-intelligence-mlp-web.json');
});
labEl('#loadWebModel').addEventListener('click',()=>labEl('#webModelFile').click());
labEl('#webModelFile').addEventListener('change',async event=>{
 const file=event.target.files?.[0];if(!file)return;
 try{const m=JSON.parse(await file.text());
 if(m.kind!=='browser'||JSON.stringify(m.shape)!=='[200,20,10,2]'||m.W?.length!==3||
  m.W.some((layer,l)=>layer?.length!==[20,10,2][l]||layer.some(row=>row.length!==[200,20,10][l]||row.some(v=>!Number.isFinite(v))))||m.B?.some((b,l)=>b.length!==[20,10,2][l]||b.some(v=>!Number.isFinite(v))))throw Error('Modelo JSON inválido ou incompatível.');
 trainedModel=m;model=m;activations=null;inputValues=null;selected=null;
 labEl('#networkState').textContent='Modelo MLP local carregado do seu JSON';labEl('#fmt').textContent='JS Float32';labEl('#modelStatus').textContent='MLP local';
 state('Modelo treinado no navegador importado. Arquivo local, sem upload.',true);setLabStatus('Modelo MLP local importado. Gere cenários e faça um novo teste.','ok');draw();
 }catch(err){setLabStatus(err.message,'bad');}
});
// A model trained locally is separate from the read-only Edge Impulse INT8 SDK.
window.forwardBrowser=function(x){return Lab.infer(model,x);};