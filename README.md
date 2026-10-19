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