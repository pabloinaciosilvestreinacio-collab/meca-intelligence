## Abrir o laboratório (sem instalar nada)

**Atenção: o código no GitHub não é automaticamente um site publicado.** Enquanto GitHub Pages não estiver habilitado, um endereço \`github.io\` pode retornar 404.

### Opção 1 — ZIP OFFLINE (funciona com duplo clique)

1. Abra [GitHub Actions — Validar Neural Lab V2](../../actions/workflows/v2.yml) e escolha a execução mais recente com marca verde.
2. Baixe o artifact **MECA-Neural-Lab-OFFLINE** (contém um ZIP).
3. Extraia **MECA-Neural-Lab-OFFLINE.zip**; mantenha a pasta \`src\` ao lado de \`index.html\`.
4. Abra \`MECA-Neural-Lab/index.html\` no **Edge ou Chrome**. O aviso **CPU local offline** confirma que o motor matemático carregou. A criação da rede e o treinamento funcionam nesse modo sem Python e sem servidor.
5. Teste 100 amostras, \`32,16\`, uma época, Adam. Depois aumente a carga se estiver estável.

O modo offline executa realmente na CPU da página e limita a geração a 5.000 amostras. O modo HTTP/HTTPS usa Web Worker, permitindo o benchmark de 10.000. Os arquivos \`src/core.js\`, \`src/lab.js\`, \`src/v2-core.js\`, \`src/v2-offline.js\`, \`src/v2-ui.js\` e \`src/v2-worker.js\` devem permanecer disponíveis. Não tente abrir o arquivo \`index.html\` diretamente na visualização de código do GitHub: isso não executa o laboratório.

### Opção 2 — Publicar site gratuito no GitHub Pages

Se o repositório ainda não tiver Pages ativo, **somente quem tem acesso às Settings do repositório** pode habilitar pela interface do GitHub:

1. Abra **Settings → Pages** no repositório.
2. Em **Build and deployment → Source**, selecione **Deploy from a branch**.
3. Escolha **main**, pasta **/(root)** e clique **Save**.
4. Aguarde a conclusão da publicação antes de acessar o endereço exibido em Settings → Pages. O site só está publicado quando o GitHub confirmar esse endereço.

Não é preciso comprar domínio. Evite publicar telemetria física privada: os arquivos importados no navegador ficam locais.

---

<div align="center">
  <img src="https://raw.githubusercontent.com/pabloinaciosilvestreinacio-collab/mecatiger-hub/main/public/Screenshot_20261004_131612_Instagram.jpg" alt="Meca Tiger" width="100" />
  <h1>MECA INTELLIGENCE · Neural Lab</h1>
  <strong>Meca Tiger FTC #32578 · Rede neural treinável direto no navegador</strong>
  <p>Neurônios e conexões reais, treinamento sintético local e inspeção experimental de um modelo Edge Impulse.</p>
</div>

## Começar

Abra `index.html` no Chrome/Edge ou hospede estes arquivos estáticos (GitHub Pages é suficiente). Não precisa instalar Python, Java, npm, cadastrar conta nem fornecer uma chave de API. Os pesos e CSVs importados ficam no navegador.

1. Clique **Gerar 900 cenários**: 600 para treino, 150 para validação e 150 para teste independente, com seed fixa 32578.
2. Clique **Treinar no navegador** e acompanhe epochs, loss e acurácia de validação. A rede **MLP 200→20→10→2** começa com pesos aleatórios e aprende via backpropagation + Adam em JavaScript.
3. Clique **Avaliar teste independente** e compare os resultados com a heurística simples. Clique **Visualizar exemplo** para observar ativações reais de um arquivo de teste.
4. Clique **Salvar modelo JSON** para guardar os pesos treinados em seu computador. **Abrir modelo JSON** permite recuperá-los posteriormente.
5. Opcional: **Exportar ZIP CSV** salva os 900 cenários para comparação em Edge Impulse. Extraia o ZIP e importe os arquivos de `training/` como Training e os de `testing/` como Testing.
6. Opcional: **Abrir ZIP / C++** para inspecionar a exportação **INT8 compilada** compatível do Edge Impulse. Isso carrega um **modelo diferente, somente leitura**: o treino web **não modifica** os pesos INT8 importados. **Abrir CSV** faz uma inferência local aproximada do modelo selecionado.

### Exatamente o que é treinado?

O novo modelo web usa 200 características derivadas de 40 leituras de IMU, encoders e potência dos motores, 20 neurônios ocultos, 10 neurônios ocultos e 2 saídas (`imu_travada`, `normal`). As entradas do treino local incluem **incrementos temporais** de sensores para melhorar a representação. **Não** são equivalentes à entrada bruta do modelo original do Edge Impulse. Treinar um modelo não retreina nem sincroniza o outro.

A visualização de conexões é baseada nos pesos atualmente carregados, e a intensidade dos neurônios vem das ativações calculadas. Sem um modelo carregado, o mapa é somente ilustrativo.

## Cenários sintéticos

A geração é determinística, com amostras independentes entre treino, validação e teste e rótulos conhecidos. Inclui curvas, giros lentos, giros com ruído, reversões, deslocamentos retos, paradas e congelamentos parciais/integrais da IMU. São **simulações didáticas**, não dados de um Control Hub ou de um robô FTC real; os modelos de sensores e cinemática são simplificados e os cenários ainda podem ser artificialmente fáceis. Um bom resultado **não comprova precisão em hardware real**.

Limitações conhecidas:
- Não mede desempenho com patinagem complexa, saturação de motores, latência real do SDK, drift de longo prazo ou sensores desconhecidos.
- A saída `imu_travada` depende dos sinais considerados no conjunto gerado; uma IMU congelada durante parada ou deslocamento reto pode ser indistinguível de operação normal.
- O teste é mantido separado da otimização, mas todos os conjuntos vêm do **mesmo gerador**, logo um teste com dados reais continua essencial.
- Uma exportação Edge Impulse compilada usa inferência JavaScript **aproximada**, não uma implementação bit a bit do runtime TFLite.
- Modelos treinados localmente são perdidos ao recarregar, a menos que você baixe o JSON.

## Estrutura

```text
index.html                 painel e visualização dos neurônios
src/core.js                geração, CSV, treino Adam/backprop, predição, avaliação e ZIP
src/lab.js                 controles do laboratório web
tests/core.test.js         testes automáticos da geração, modelo e exportação
.github/workflows/ci.yml  validação com Node.js
```

## Testes automáticos

Com Node.js 20+:

```bash
node --test tests/*.test.js
```

Também é possível simplesmente abrir o HTML para uso, sem Node.

## Publicação no GitHub Pages

1. Crie um repositório GitHub chamado `meca-intelligence`, sem adicionar o ZIP do SDK, dados reais ou tokens.
2. Envie **somente os arquivos deste projeto** para a branch `main` (com `index.html` na raiz).
3. Em **Settings → Pages → Build and deployment**, selecione **Deploy from a branch → main → /(root)**. Aguarde a publicação.
4. O endereço esperado será `https://<usuario>.github.io/meca-intelligence/`, depois da ativação do Pages.

O GitHub Pages é hospedagem estática: não oferece execução segura de código Java arbitrário nem permite retreinar o modelo do SDK Edge Impulse no servidor.

## Licença e privacidade

O código **original deste laboratório web** é disponibilizado sob MIT (consulte `LICENSE`). Isso **não** altera os direitos sobre modelos/SDK exportados por terceiros. **Não publique o ZIP exportado do Edge Impulse**, pesos proprietários, código privado da equipe ou CSVs coletados no robô sem conferir permissões, privacidade e licença. Para manter os dados privados, a página processa os arquivos localmente, sem serviço backend.

Não use as previsões para controle direto e automático de motores antes de validação e controles físicos de segurança.

---

## MECA Intelligence Hybrid — diagnóstico offline MIH-V1 (experimental)

O laboratório de rede neural **acima continua disponível sem modificações**. Foi adicionado um núcleo independente em Python para organizar ensaios do Mequinha e aplicar diagnósticos matemáticos explicáveis. **Ele não comanda motores, não lê o Control Hub, não altera o Meca Sim e não valida a rede neural no robô físico.**

### Comece pelo diagnóstico

Requisito: Python 3.11+ (somente biblioteca padrão).

```bash
python -m unittest discover -s tests -p 'test_mih.py' -v
python -m mih --help
```

No Meca Sim, clique em **Exportar CSV** depois da execução e guarde `meca-sim-trajetoria.csv`. No diretório raiz do MIH:

```bash
python -m mih import meca-sim-trajetoria.csv \
  --source simulation \
  --run-id ensaio-sim-001 \
  --recorded-at 2026-10-19T20:00:00-03:00 \
  --robot-config mequinha-experimental \
  --code-version unknown \
  --experiment-id avaliacao-giro \
  --report local/ensaio-sim-001.md

python -m mih list
python -m mih analyze ensaio-sim-001 --report local/ensaio-revisado.md
python -m mih compare ensaio-sim-001 ensaio-sim-002
```

**Substitua a data e os identificadores do exemplo pelos valores efetivos.** `--recorded-at` representa uma informação fornecida pelo operador; o software não a adivinha.

Os dados importados ficam no banco SQLite local `local/mih.sqlite3`, ignorado pelo Git. O relatório registra [R] observações, [H] hipóteses e [P] pendências. Falta de canais gera `indeterminate`; o yaw exportado da trajetória é o ângulo real **simulado**, nunca uma leitura da IMU do Hub. O resultado da rede neural no navegador continua sendo exclusivamente experimental.

Documentação: [contrato de dados MIH-V1](docs/MIH-DATA-CONTRACT-v1.md) · [auditoria M.E.C.A. e limitações](docs/MIH-ENGINEERING-AUDIT.md).

### Estado das etapas

- Etapa 0: contrato e auditoria registrados.
- Etapa 1: importação CSV/JSON, validação, SQLite, regras e CLI (sujeitos a CI).
- Etapa 2: laboratório neural existente preservado; **nenhum modelo validado em hardware**. Treinamento físico suspenso por falta de dados rotulados.
- Etapa 3: comparação descritiva de execuções, sem otimização automática de OpModes.
- Etapa 4: adaptador de gravações Meca Sim no importador; testes de bancada e coleta automática ainda pendentes.

**Atenção:** não versione CSVs reais sem revisar privacidade e consentimento. Limiares são didáticos e não certificados.

### Avaliação supervisionada segura (opcional)

O subcomando `ml-check` exige identificador de sessão (`--session-id` na importação), rótulos revisados, sinais suficientes, ao menos 12 execuções por classe e sessões independentes; `real` exige `label_provenance=verified_measurement`. Ele mostra motivos de bloqueio antes de qualquer treino.

```bash
python -m mih ml-check --domain real
# Somente após a prontidão, instale opcionalmente: pip install scikit-learn
python -m mih ml-evaluate --domain real
```

`ml-evaluate` compara preditor majoritário, regressão logística e Random Forest, com **holdout agrupado por sessão**, métricas por classe e sem salvar pesos automaticamente. Também permite `--domain synthetic_lab`, sempre identificado como experimental; não converta esse resultado em afirmação sobre o Mequinha físico. O laboratório MLP existente no navegador permanece independente.


---

# Neural Lab V2 — laboratório configurável (experimental)

A V2 é uma **nova área** na mesma aplicação, abaixo do laboratório legado. A rede existente, a importação Edge Impulse e o MIH-V1 Python foram preservados.

A V2 usa um **Web Worker CPU**; para funcionar, abra com servidor HTTP local (não use duplo clique \`file://\`):

\`\`\`bash
python -m http.server 8000
\`\`\`

Abra \`http://localhost:8000\` e localize o painel **MECA NEURAL LAB V2**.

1. Escolha camadas ocultas (\`20,10\`, \`32,16\`, \`64,32,16\` ou \`128,64,32\`) e inicialize os pesos.
2. Defina até 10.000 exemplos sintéticos, seed e divisão de treino/validação/teste e gere os dados.
3. Configure épocas, batch size, taxa de aprendizado, Adam/SGD e parada antecipada; treine no Worker.
4. Use o mapa de pesos para zoom, inspeção de bias e ativações. Avalie o conjunto reservado, exporte JSON e histórico.
5. Importe telemetria MIH-V1, CSV ou exportação do Meca Sim. O modelo só aceita inferência se possuir 40 amostras e todos os cinco canais requeridos.
6. Use o tutorial de oito aulas e consulte o módulo de processamento. GPU e VM remota **não** são recursos disponíveis.

\`\`\`bash
node --test tests/core.test.js tests/v2.test.js
python -m unittest discover -s tests -p 'test_*.py'
\`\`\`

Os benchmarks CPU 1k/5k/10k e o teste Chromium/Playwright estão nos jobs de [GitHub Actions](https://github.com/pabloinaciosilvestreinacio-collab/meca-intelligence/actions). Confira suas execuções efetivamente aprovadas; um teste unitário, sozinho, não valida a interface.

[Arquitetura, fontes, limitações, contrato e evidências V2](docs/MIH-NEURAL-LAB-V2.md)

**Importante:** nem cenários sintéticos nem um CSV sem procedência autenticada confirmam o desempenho neural no Mequinha real.

## Verificações e evidências da V2

[Consultar matriz de aceitação, benchmarks medidos, capturas Chromium e riscos](docs/MIH-V2-VALIDATION.md).

A validação física segue pendente; a V2 não controla motores e não adiciona GPUs ou VMs fictícias.
