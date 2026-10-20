# Auditoria M.E.C.A. — 2026-10-19

## M — Mapear

- Repositório de destino já existente e **público**: `meca-intelligence`. Contém UI estática com laboratório neural de 900 amostras sintéticas, MLP 200→20→10→2, testes JS e documentação MIT. **Nenhum desses arquivos foi sobrescrito pelo núcleo MIH.**
- `meca-sim` 0.3-rc2: simulador local Python/Java/JS com CSV de trajetória e API `/api/recording`. A API exporta `time_s,x_m,y_m,yaw_deg` e `*_ticks`. Esse `yaw_deg` é a pose verdadeira simulada, não a leitura IMU usada pelo OpMode. Não contém comandos por intervalo, tensão, metadados de sessão ou rótulos confirmados.
- Diagnósticos atuais do simulador detectam falhas injetadas; não são evidência de uma falha física.
- Robô real: são necessários registros da IMU e dos encoders com relógio e convenção de sinal, potência comandada, tensão quando disponível, configuração e commit do OpMode. Não existe, nesta auditoria, dataset físico rotulado e independente.

## E — Explorar: alternativas

| Abordagem | Custo/dependências | Vantagens | Limitações | Decisão |
|---|---|---|---|---|
| Regras determinísticas + estatística descritiva | Python stdlib | Auditável, fácil reproduzir, funciona com poucos dados | Limiar depende da condição; falso positivo | **Implementado no MVP** |
| Regressão logística/Random Forest (scikit-learn) | Dependência opcional + dados rotulados | Baseline mensurável, interpretação parcial | Vazamento entre janelas e domínio sim/real | **Adiar treinamento** |
| Pequena MLP / Edge Impulse | Treino, calibração e manutenção | Pode aprender padrões temporais sutis | 100% em dados do mesmo gerador não valida hardware; baixa amostragem física | **Preservado experimento web; não promover ao robô** |

Fontes revisadas (documentação, não modelos executados):

- REV IMU: https://docs.revrobotics.com/duo-control/sensors/i2c/imu/orientating-the-imu
- Edge Impulse data acquisition: https://docs.edgeimpulse.com/reference/data-ingestion/data-acquisition-format
- Edge Impulse CSV: https://github.com/edgeimpulse/tool-data-collection-csv
- scikit-learn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

Licenças dos pacotes, versões de modelos externos e datasets comparáveis: **não avaliados por execução**. Nenhum checkpoint externo foi carregado. Os resultados de comparações são conceituais, não benchmarks.

## C — Concretizar

- Núcleo `mih/core.py`: contrato MIH-V1, CSV/JSON, rejeição de valores inválidos, detecções observacionais, SQLite com ID imutável, comparação descritiva.
- Interface `python -m mih`: importação, análise, relatórios Markdown, listagem, comparação.
- Testes com fixtures didáticas, separados dos testes de hardware.
- `meca-sim`, `src/core.js`, `src/lab.js` e os OpModes permanecerão intactos.

## A — Avaliar / riscos

1. **Crítico:** ainda não existe evidência suficiente para validar o diagnóstico no hardware; nenhuma acurácia de bancada deve ser declarada.
2. **Alto:** metadata `source` declarada por operador, sem atestado criptográfico.
3. **Alto:** comandos e leitura IMU não vêm do exportador padrão do simulador; diagnósticos ausentes são classificados inconclusivos.
4. **Alto:** regras por limiar não generalizam automaticamente entre configurações.
5. **Médio:** SQLite local é monoestação (equipe deve fazer backup controlado).
6. **Médio:** análise em lote/CLI não é ferramenta de segurança operacional em tempo real.

### Próximo experimento real (proposta, não teste realizado)

Capturar sessões distintas com `time_ms`, potências, encoders, IMU, tensão, montagem e versão imutável do código. Protocolar `ensaio`, `falha observada`, `resultado rotulado por avaliador` e condições; testar 3–5 repetições sob condições controladas. Revisar thresholds usando **sessões** independentes. Só então reavaliar classificadores e aprendizado.

**Decisão M.E.C.A.:** continuar com diagnósticos determinísticos e coleta estruturada; suspender promessas de validação neural física.

## Continuação condicional da Etapa 2

Implementado protocolo de prontidão do aprendizado (`mih/ml.py`). Exige rótulos controlados, amplitude dos sinais, execuções independentes, `session_id` e classes suficientes. Quando o critério é atendido, compara modelo majoritário, regressão logística e Random Forest com divisão por sessão; dependência scikit-learn opcional. **Nenhum modelo foi considerado treinado ou validado fisicamente nesta auditoria.** Regressões com amostras artificiais testam apenas a implementação das regras de prontidão.
