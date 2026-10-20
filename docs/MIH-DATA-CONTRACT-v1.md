# Contrato MIH-V1 — MECA Intelligence Hybrid

Status: **experimental**, 19/10/2026. Formato canônico JSON UTF-8. Versão obrigatória: `MIH-V1`.

## Envelope da execução

Campos **obrigatórios**: `schema_version`, `run_id` (único, até 100 caracteres [A-Za-z0-9_.-]), `source`, `timestamp` ISO 8601 com fuso horário explícito, `robot_config_id`, `code_version`, `experiment_id`, `samples` (lista não vazia).

`source`: `real` (operador declarou coleta física), `simulation` (Meca Sim), `synthetic_lab` (gerador didático/Edge Impulse sintético). A origem é **declarada**, não autenticada.

Opcionais: `result_label` (ou null), `label_provenance` = `unlabeled`, `operator`, `synthetic_injected`, `verified_measurement`; `termination_reason` = `unknown`, `completed`, `timeout`, `stopped`, `error`; `notes`, `import_format`, `source_sha256`.

Valores desconhecidos de configuração ou commit precisam ser explicitamente marcados `unknown`. Nunca inferir identificação a partir de nome de arquivo.

## Colunas de cada amostra

| Coluna | Unidade | Disponibilidade e significado |
|---|---|---|
| `time_ms` | ms | **Obrigatória**, crescente, >= 0 |
| `motor_me_command`, `motor_md_command` | potência [-1, 1] | Opcional; comando lógico, não torque real |
| `encoder_me_ticks`, `encoder_md_ticks` | ticks | Opcional; requer convenção de sinal para inferência cinemática |
| `imu_yaw_deg` | graus | Opcional; **leitura da IMU**, não yaw da pose simulada |
| `battery_voltage_v` | V | Opcional, 0–35 |
| `target_distance_m`, `observed_distance_m` | m | Opcionais, só comparar quando ambos tiverem mesma definição física |
| `target_yaw_deg` | graus | Opcional |
| `pose_x_m`, `pose_y_m` | m | Opcionais; posição simulada, não medição física |
| `pose_yaw_deg` | graus | Opcional; orientação real do mundo simulado, **não IMU** |

Números precisam ser finitos, sem booleans ou textos; null é aceito para canais opcionais. Amostras têm limite de 200.000; arquivos importados têm limite de 32 MiB. Formato inválido causa erro explícito.

## Adaptadores disponíveis

**Meca Sim 0.3-rc2:** botão "Exportar CSV"/API `/api/recording`, colunas `time_s,x_m,y_m,yaw_deg,ME_ticks,MD_ticks,...`. Conversão: segundos × 1000 → `time_ms`; `yaw_deg` → `pose_yaw_deg`; `ME_ticks` e `MD_ticks` → encoders. **Não** atribuir `yaw_deg` à IMU. Nem potência, nem tensão são incluídas; permanecerão ausentes. Formato JSON bruto `{"columns":[...],"rows":[...]}` da API também aceito. A exportação "resultado.json" contém estado, não série temporal; é recusada como série.

**Laboratório neural já existente:** CSV `timestamp,imu_yaw_deg,encoder_md_ticks,encoder_me_ticks,potencia_md,potencia_me`; a janela de 40 leituras pode ser importada como uma execução `synthetic_lab`. Sempre registrar o rótulo e sua origem de forma explícita.

**MIH CSV:** cabeçalho com `time_ms` e qualquer subconjunto reconhecido das colunas opcionais. Para construir o envelope, parâmetros CLI obrigatórios identificam sessão, origem e versão do código.

**MIH JSON canônico:** importação valida o envelope e o preserva; não substitui o `source` declarado, e não simula ausência de campos.

## Estratégia de diagnóstico

Anomalias representam **sintomas**: IMU sem avanço angular durante giro comandado, encoder sem progresso sob potência, assimetria de encoders sob comandos semelhantes, desvio entre distância alvo e observada, timeout declarado e lacunas de amostragem. Os limiares são heurísticos **não calibrados** para o Mequinha. Ausência de canal essencial → diagnóstico `indeterminate` quando não houver outro evento. Nenhum diagnóstico confirma dano em motor/encoder/IMU.

## Exemplo mínimo

```json
{
  "schema_version": "MIH-V1",
  "run_id": "ensaio-001",
  "source": "simulation",
  "timestamp": "2026-10-19T18:00:00-03:00",
  "robot_config_id": "unknown",
  "code_version": "unknown",
  "experiment_id": "exploratorio",
  "label_provenance": "unlabeled",
  "samples": [{"time_ms": 0}, {"time_ms": 50}]
}
```

## Evolução

Alterações incompatíveis criarão `MIH-V2` e migrador explícito. Preservar JSON bruto, SHA256 e identificação da execução no banco SQLite local.

## Treinamento supervisionado — metadata estendida

`session_id` é opcional na importação comum, mas **obrigatório para elegibilidade no treinamento**. Deve identificar sessão/coleta independente; janelas do mesmo ensaio não podem ser divididas entre treino e teste. Conjuntos com sinais duplicados são descartados da avaliação. Para treinamento físico, `label_provenance=verified_measurement` requer revisão humana real das evidências; o software apenas verifica a declaração, não autentica sua veracidade. Classes aceitas para a tarefa binária: `normal` e categorias conhecidas de anomalias. Um modelo não treinado por falta de dados é uma decisão correta, não uma falha do sistema.
