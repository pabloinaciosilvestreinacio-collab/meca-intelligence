# MECA Neural Lab V2 — arquitetura e registro técnico

Status: **experimental em branch até aprovação dos gates**. Projeto Meca Tiger FTC #32578, temporada BIOBUZZ 2026–2027.

## Baseline auditado

- Branch inicial main, SHA: b6aa7c6c444ca5c914fcf574a8b7e28ae977e938.
- CI main inicialmente aprovada: workflow Validar Meca Intelligence.
- Laboratório legado: src/core.js (MLP 200→20→10→2, geração sintética, Adam), src/lab.js, visualizador INT8 experimental dentro do index.html, tests/core.test.js.
- MIH-V1 Python: mih/core.py, mih/ml.py, mih/cli.py, testes e contrato em docs/MIH-DATA-CONTRACT-v1.md.
- Alterações V2 não modificam Meca Sim nem OpModes do robô.

## Arquitetura efetiva

| Componente | Responsabilidade | Observação |
|---|---|---|
| src/v2-core.js | Configuração validada, pesos, biases, forward, backprop, Adam/SGD, métricas, importação | CPU JavaScript puro, classes reais imu_travada / normal |
| src/v2-worker.js | Gerar dados, treinar, avaliar e medir cargas fora da UI | Web Worker de CPU, sem GPU |
| src/v2-ui.js | Painel real, grafo com pesos, controles, zoom, telemetria, tutoriais, exportação | Não é um mockup; processa mensagens do worker |
| tests/v2.test.js | Arquiteturas, regressão, gradiente por diferenças finitas, contratos | Node.js |
| tests/v2-browser.test.js | Ações em Chromium, downloads, erro de importação, responsividade | Playwright |
| tests/v2-benchmark.js | Medição real 1k, 5k e 10k amostras, CPU Node GitHub Actions | Tempo e heap medidos no runner |
| mih/* | Pipeline independente diagnóstico offline MIH-V1 | Inalterado pela V2 |

### Modelo

\`shape=[200,...hidden,2]\`; 1 a 4 camadas ocultas de 2 a 128 neurônios, limitado a 60.000 parâmetros. \`ReLU\` e \`tanh\` são ativadores ocultos; última camada aplica softmax estável. \`W[l][j][i]\` conecta entrada \`i\` ao neurônio \`j\`; bias por neurônio. Serialização: JSON \`kind:meca-v2\`, \`version:2\`, dimensões, labels, ativação, pesos, bias, seed e histórico treinado. Importador preserva suporte a \`kind:browser\` com formato legado 200→20→10→2.

Entradas continuam **40 amostras × 5 canais a 20 Hz**, conforme gerador sintético original. A UI não oferece redimensionar a janela nem classes adicionais porque o pipeline não fornece treinamento compatível; aumentar o número de neurônios ocultos **não** aumenta o total de amostras. \`MAX_PARAMS\` evita explosão de memória em configurações extremas.

### Treinamento

- Learning rate, épocas, batch, seed, Adam/SGD e parada antecipada (patience) operam no treino real.
- Validação por época mede perda/correção sem tocar no conjunto de teste até clique separado.
- Cancelamento é cooperativo e usa yield ao loop de eventos no Worker. A UI preserva modelo anterior ao treino incompleto.
- \`feature()\` reproduz diferenças temporais e clipping do laboratório original para preservar compatibilidade da MLP antiga.
- Um novo treino a partir de pesos treinados é confirmado. Reinicializar pede confirmação se houver pesos treinados ainda não exportados.
- Dados sintéticos se originam do mesmo gerador \`MecaCore.createDataset\` (seed registrada); splits distintos **não comprovam generalização de domínio físico**.

### Visualização

Canvas 2D em grafite/laranja, pesos reais \`W\`, sinal positivo/negativo, bias e ativação reais quando calculada. Adapta o número de camadas e neurônios à topologia; limite de 80 neurônios exibidos por camada e 6.500 linhas desenhadas, com contagem visível e real. Zoom, pan, espessura controlada por magnitude, filtro de densidade e opacidade apenas afetam desenho, **não editam pesos**. Não apresenta GPU fictícia. Personalização e zoom permanecem efêmeros nesta primeira iteração, não sincronizados entre dispositivos.

## Importação e privacidade

A UI processa arquivos localmente, até 8 MiB. Importa JSON \`MIH-V1\`, CSV MIH, CSV do Neural Lab e trajetórias CSV/JSON \`/api/recording\` do Meca Sim. Exige origem declarada e mostra prévia, cabeçalhos/sinais ausentes, timestamps crescentes, potência [-1,1] e compatibilidade com 40 leituras 20 Hz. Não reamostra e não inventa IMU a partir do yaw do simulador. **Limitação:** a validação do navegador é voltada ao vetor neural, não substitui a validação completa do schema no módulo Python.

Dados físicos não deixam navegador por padrão. O título "real" é informação do operador, nunca confirmação criptográfica. Não implementar upload automático.

## Ambientes de processamento

**Implementado:** navegador CPU, Web Worker real; execução assíncrona sem bloquear a interface. Node.js CPU em GitHub Actions para benchmarks. **Não implementado:** backend Python de treino, máquina virtual externa e WebGPU/GPUs; sem menus falsos ou simulações desses ambientes. A CI não comprova o desempenho do PC/celular do usuário.

## Benchmarks e evidências

O job \`benchmarks\` mede **realmente** geração e uma época de treino para 1.000, 5.000, 10.000 amostras no runner Linux/Node; grava \`v2-benchmarks.json\` como artefato da execução. O painel possui benchmark complementar no Worker, sempre distinguindo tempos do navegador dos tempos do runner. Os valores de tempo só podem ser publicados quando lidos da execução real aprovada.

O job \`browser\` roda Chromium via Playwright e produz \`v2-desktop.png\` e \`v2-mobile.png\` como capturas reais, após executar o software. Se a CI falhar, **não** afirmar validação E2E.

## Testes de aceitação e limitações

- Regressão de \`tests/core.test.js\` e testes Python \`test_*.py\` continuam obrigatórios.
- Testes matemáticos devem passar com inicialização em quatro arquiteturas, dimensões, comparação de derivadas por diferenças finitas, treino real, softmax, serialização e rejeição de casos inválidos.
- E2E deve alterar arquitetura, treinar Worker, renderizar topologia, importar dados, obter predição e exercitar mobile/download.
- Medições físicas do Mequinha: **PENDENTES — SEM DATASET REAL DO MEQUINHA**.
- Não há validação independente de precisão com dataset físico, nem avaliação externa de especialistas/modelos autônomos. Auditoria adversarial manual testou incompatibilidades, arquivos incompletos e falsas alegações de aceleração.

### Matriz de decisão M.E.C.A.

Continuar incrementos verificados. **Não fazer merge** enquanto os gates de software falharem. Não promover classificações a ferramenta competitiva sem sessões reais verificadas.

## Fontes externas avaliadas (sem copiar código)

| Referência | Licença conhecida | Lições | Decisão |
|---|---|---|---|
| [TensorFlow Playground](https://github.com/tensorflow/playground) | Apache-2.0 | Manipulação pedagógica de camadas e pesos com interface interativa | Inspirou separação didática, sem reutilização de código |
| [Netron](https://github.com/lutzroeder/netron) | MIT | Grafo de modelos reais; inspeção de operações e formatos | Não importado: formatos proprietários/ONNX fora do escopo |
| [TensorFlow.js](https://github.com/tensorflow/tfjs) | Apache-2.0 | CPU/WASM/WebGPU e treino em navegador | Alternativa futura após medir custos |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Backends de inferência real e requisitos explícitos de WebGPU | Não instalado; GPU só pode surgir após implementação |
| [MDN Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) | Documentação MDN, licença própria | Tarefas fora da thread principal, mensagens por clone estruturado | Usado como referência para CPU Web Worker |
| [D3.js](https://d3js.org/) | ISC | Visualização e layout de grafos | Canvas 2D simples escolhido por dependência zero |

Bibliotecas citadas não foram automaticamente instaladas ou usadas como motores. Comparação conceitual, não benchmark externo de performance.

## Guia de uso local

O laboratório legado pode funcionar como arquivo estático, mas a V2 Worker usa origem HTTP. Execute:

\`\`\`bash
python -m http.server 8000
\`\`\`

Abra \`http://localhost:8000\`, navegue até *Neural Lab V2*, inicialize uma arquitetura, gere dados e treine. Use \`node --test tests/core.test.js tests/v2.test.js\` e \`python -m unittest discover -s tests -p 'test_*.py'\` para auditoria offline.
