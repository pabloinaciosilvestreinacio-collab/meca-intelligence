# V2 — Matriz de verificação, benchmarks e riscos

Data da auditoria: 2026-10-19/20. Repositório: Meca Tiger FTC #32578. **Não confundir CI, simulação e validação no Mequinha físico.**

## Provas reproduzíveis

- SHA de base da main: \`b6aa7c6c444ca5c914fcf574a8b7e28ae977e938\`.
- CI com testes matemáticos e worker: [Validar Neural Lab V2](https://github.com/pabloinaciosilvestreinacio-collab/meca-intelligence/actions/workflows/v2.yml).
- CI preexistente Node + MIH-V1 Python: [Validar Meca Intelligence](https://github.com/pabloinaciosilvestreinacio-collab/meca-intelligence/actions/workflows/ci.yml).
- Evidência de benchmarks efetivamente medida: [execução 39348931569](https://github.com/pabloinaciosilvestreinacio-collab/meca-intelligence/actions/runs/39348931569), artefato \`mih-v2-benchmarks\`.
- Evidência visual de Chromium desktop/mobile após mover V2 ao topo: [execução 39349696352](https://github.com/pabloinaciosilvestreinacio-collab/meca-intelligence/actions/runs/39349696352), artefato \`neural-v2-chromium-screenshots\`. As capturas são renderizações efetivas do navegador em CI, **não mockups**.

## Benchmarks reais: CPU Linux x64 no GitHub Actions

Executor reportado: \`GitHub Actions Linux Node CPU\`. Node v22.23.3, rede 200→20→10→2, 1 época, batch size 64, seed 32578.

| Amostras | Tempo geração | Tempo treino | Heap usado observado* | Concluiu | Loss validação |
|---:|---:|---:|---:|---|---:|
| 1.000 | 80 ms | 340 ms | 10,38 MiB | sim | 0,384672 |
| 5.000 | 344 ms | 447 ms | 30,96 MiB | sim | 0,046973 |
| 10.000 | 648 ms | 752 ms | 72,89 MiB | sim | 0,024536 |

* Heap do processo consultado no final de cada lote; **não** é pico de RSS nem consumo total do navegador. Resultados dependem fortemente do executor e do número de parâmetros. Este benchmark usa **apenas uma época** e não mede generalização física. Números extraídos do log real da execução mencionada; outras execuções podem ter tempos diferentes.

## Evidências funcionais esperadas para o commit final

| Requisito | Como é testado | Classificação |
|---|---|---|
| Editor de topologia e quatro tamanhos de rede | Node: formatos, forward, gradientes, exportação; Chromium: troca real de arquitetura | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| Backprop/Adam/SGD e softmax estável | Diferenças finitas, atualização real de pesos, loss finita e métricas | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| Dataset 1k/5k/10k | Benchmark GitHub Linux CPU e testes de geração reproduzível | FUNCIONAL E VERIFICADO para configurações benchmarkadas |
| Canvas de pesos reais, filtros, zoom, pan, preferências | Chromium: grafo após selecionar neurônio e screenshot real | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| Modelo JSON e comparação de dois modelos | Chromium: download, reimportação, avaliação dos mesmos dados | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| Importação e prévia MIH-V1, CSV e simulação | Testes unitários e Chromium com arquivos sintéticos bem-formados/malformados | FUNCIONAL E VERIFICADO como software |
| Relatório M.E.C.A. | Exportação real no Chromium; campos ausentes ficam null | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| Tutorial em português de oito aulas | Interface de ações documentadas | IMPLEMENTADO, MAS SEM ESTUDO FORMAL DE USABILIDADE |
| CPU Web Worker | Execução real, progresso, treino e cancelamento | FUNCIONAL E VERIFICADO (ao aprovar CI) |
| GPU/WebGPU | Backend não implementado | NÃO IMPLEMENTADO, não exposto como operacional |
| Backend Python de treino/VM externa | Não provisionado, sem autenticação/infraestrutura | NÃO IMPLEMENTADO; não exposto como operacional |
| Janela temporal variável, reamostragem | Pipeline atual exige 40×5 a 20 Hz | NÃO IMPLEMENTADO, explicitamente bloqueado |
| Edição manual de peso matemático | Recurso experimental investigado, não entregue | NÃO IMPLEMENTADO (para evitar métricas antigas enganadoras) |
| Validação no Mequinha físico | Sem dataset físico autêntico | **BLOQUEADO — VALIDAÇÃO FÍSICA PENDENTE** |

## Auditoria adversarial e correções

1. **Bug:** checagem de peso único parecia falhar embora outros pesos fossem atualizados. **Ação:** teste compara matrizes completas.
2. **Bug:** cálculo de limite aceitava rede grande prevista para bloqueio. **Ação:** limite conservador de 60.000 parâmetros.
3. **Bug:** conclusão do Worker não habilitava o botão de treino. **Ação:** sincronização de estado corrigida e coberta por Chromium.
4. **Bug:** workflow antigo tentava executar teste Playwright sem a dependência. **Ação:** job próprio de navegador com instalação isolada, mantendo os testes matemáticos antigos.
5. **Bug visual:** filtros podiam ocultar todas as arestas do neurônio selecionado. **Ação:** conexões incidentes são mostradas pelo peso real, independentemente do filtro; regressão de navegador.
6. **Risco:** dataset com timestamp ausente poderia ser convertido implicitamente para zero. **Ação:** rejeição explícita; validação de metadados MIH-V1.
7. **Risco metodológico:** cálculo da baseline sobre o conjunto de teste antes do fim da validação. **Ação:** baseline é calculada na validação, teste fica reservado para uso explícito.
8. **Regressão:** laboratório V1 e MIH-V1 não podem ser removidos. **Ação:** CI conserva ambos e V2 usa namespaces próprios.

## Riscos e decisões

- **Alto:** a origem \`real\` no navegador é uma declaração do operador, não prova criptográfica de coleta. Se um arquivo não tiver proveniência, **não** declare validação física.
- **Alto:** o simulador atual não exporta yaw IMU nem potência como sinais da gravação padrão; a V2 bloqueia inferência para essas amostras incompletas.
- **Alto:** redes grandes + muitas épocas no browser podem consumir CPU/memória e aquecer dispositivos; benchmarks medidos em Node Linux não dão garantia em celular.
- **Médio:** dataset de treinamento/validação/teste do mesmo gerador pode favorecer aprendizagem de artefatos sintéticos.
- **Médio:** comparação reiterada sobre teste reservado pode orientar hiperparâmetros indevidamente; a interface contém aviso.
- **Médio:** as telas mobile são verificadas em Chromium emulado, não em aparelhos Android físicos.
- **Médio:** importação do navegador valida o vetor para inferência e contrato mínimo, enquanto a validação completa persistente de experimento permanece no MIH-V1 Python.

## Decisão M.E.C.A.

**Continuar o laboratório V2 de CPU local após validação da CI; preservar componentes V1; proibir alegação de validação física.** Os backends não implementados permanecem ausentes; nenhuma cobrança ou VM foi provisionada. A próxima prioridade, **não iniciada automaticamente**, é produzir telemetria real controlada do Mequinha e separar sessões por execução antes de treinar/avaliar o classificador com dados físicos.
