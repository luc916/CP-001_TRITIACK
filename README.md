# ALUNO - LUCAS KENZO NISHIWAKI / RM 561325 / TURMA 1CCR

## INSTRUÇÕES DA ENTREGA:
**1)** A atividade pode ser desenvolvida em grupo.
**2)** Apenas um integrante submete a atividade.
**3)** Enviar apenas o link do repositório.

# Individual Household Electric Power Consumption

**Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)  
**Doado em:** 29/08/2012  
**Período de medição:** Dezembro de 2006 a Novembro de 2010 (47 meses)  
**Local:** Sceaux, França (7 km de Paris)  
**Número de instâncias:** 2.075.259  
**Número de variáveis:** 9  
**Tipo de dados:** Multivariado, Série Temporal  
**Tarefas associadas:** Regressão, Clustering  

## Descrição
Medições de consumo elétrico em uma residência com taxa de amostragem de 1 minuto ao longo de quase 4 anos. Diferentes grandezas elétricas e submedições específicas estão disponíveis.  

**Observações importantes:**
- Valores ausentes representam cerca de 1,25% das linhas.  
- A energia ativa não registrada pelos sub-meters 1, 2 e 3 pode ser calculada por:  
  `(global_active_power * 1000 / 60) - sub_metering_1 - sub_metering_2 - sub_metering_3`  
- Os timestamps estão completos, mas alguns valores de medição estão ausentes (representados pela falta de valor entre dois pontos e vírgulas consecutivos).

## Variáveis
| Nome da Variável           | Tipo        | Descrição                                                                 | Unidade | Valores ausentes |
|----------------------------|------------|---------------------------------------------------------------------------|--------|----------------|
| Date                       | Feature    | Data do registro                                                           | dd/mm/yyyy | Não           |
| Time                       | Feature    | Hora do registro                                                           | hh:mm:ss  | Não           |
| Global_active_power        | Feature    | Potência ativa global média por minuto                                     | kW     | Não           |
| Global_reactive_power      | Feature    | Potência reativa global média por minuto                                   | kW     | Não           |
| Voltage                    | Feature    | Tensão média por minuto                                                    | V      | Não           |
| Global_intensity           | Feature    | Intensidade global média por minuto                                        | A      | Não           |
| Sub_metering_1             | Feature    | Consumo sub-meter 1 (cozinha)                                             | Wh     | Não           |
| Sub_metering_2             | Feature    | Consumo sub-meter 2 (lavanderia)                                          | Wh     | Não           |
| Sub_metering_3             | Feature    | Consumo sub-meter 3 (aquecedor/AC)                                        | Wh     | Não           |

## Arquivo do dataset
- `household_power_consumption.txt` (126,8 MB)  
Formato CSV delimitado por ponto e vírgula (`;`), com valores ausentes representados por `?`.





# Appliances Energy Prediction

**Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
**Doado em:** 14/02/2017  
**Período de medição:** Aproximadamente 4,5 meses  
**Local:** Casa de baixo consumo energético, Bélgica (dados de sensores internos + estação meteorológica)  
**Número de instâncias:** 19.735  
**Número de variáveis:** 28  
**Tipo de dados:** Multivariado, Série Temporal  
**Tarefas associadas:** Regressão  

## Descrição
Dados experimentais usados para criar modelos de previsão do consumo energético de eletrodomésticos em uma residência de baixo consumo.  
- Condições de temperatura e umidade monitoradas com sensores ZigBee.  
- Dados internos coletados a cada 10 minutos (média dos dados transmitidos a cada 3,3 minutos).  
- Dados meteorológicos externos obtidos da estação do aeroporto de Chievres, Bélgica.  
- Inclui duas variáveis aleatórias (`rv1`, `rv2`) para testes de modelos de regressão.  

## Variáveis
| Nome da Variável | Tipo        | Descrição                                                                 | Unidade | Valores ausentes |
|-----------------|------------|---------------------------------------------------------------------------|--------|----------------|
| date            | Feature    | Data e hora do registro                                                    | yyyy-mm-dd hh:mm:ss | Não |
| Appliances      | Target     | Consumo de energia de eletrodomésticos                                     | Wh     | Não |
| lights          | Feature    | Consumo de energia das luzes da casa                                       | Wh     | Não |
| T1              | Feature    | Temperatura na cozinha                                                     | °C     | Não |
| RH_1            | Feature    | Umidade na cozinha                                                         | %      | Não |
| T2              | Feature    | Temperatura na sala                                                        | °C     | Não |
| RH_2            | Feature    | Umidade na sala                                                            | %      | Não |
| T3              | Feature    | Temperatura na lavanderia                                                  | °C     | Não |
| RH_3            | Feature    | Umidade na lavanderia                                                     | %      | Não |
| T4              | Feature    | Temperatura no escritório                                                  | °C     | Não |
| RH_4            | Feature    | Umidade no escritório                                                     | %      | Não |
| T5              | Feature    | Temperatura no banheiro                                                   | °C     | Não |
| RH_5            | Feature    | Umidade no banheiro                                                       | %      | Não |
| T6              | Feature    | Temperatura externa (lado norte)                                           | °C     | Não |
| RH_6            | Feature    | Umidade externa (lado norte)                                              | %      | Não |
| T7              | Feature    | Temperatura na sala de passar roupa                                       | °C     | Não |
| RH_7            | Feature    | Umidade na sala de passar roupa                                           | %      | Não |
| T8              | Feature    | Temperatura no quarto de adolescente 2                                    | °C     | Não |
| RH_8            | Feature    | Umidade no quarto de adolescente 2                                        | %      | Não |
| T9              | Feature    | Temperatura no quarto dos pais                                             | °C     | Não |
| RH_9            | Feature    | Umidade no quarto dos pais                                                 | %      | Não |
| T_out           | Feature    | Temperatura externa (estação Chievres)                                     | °C     | Não |
| Pressure        | Feature    | Pressão externa (estação Chievres)                                         | mmHg   | Não |
| RH_out          | Feature    | Umidade externa (estação Chievres)                                        | %      | Não |
| Windspeed       | Feature    | Velocidade do vento (estação Chievres)                                     | m/s    | Não |
| Visibility      | Feature    | Visibilidade (estação Chievres)                                           | km     | Não |
| Tdewpoint       | Feature    | Ponto de orvalho (estação Chievres)                                       | °C     | Não |
| rv1             | Feature    | Variável aleatória 1                                                       | N/A    | Não |
| rv2             | Feature    | Variável aleatória 2                                                       | N/A    | Não |

## Arquivo do dataset
- `energydata_complete.csv` (11,4 MB)  
Formato CSV delimitado por vírgula, sem valores ausentes.  
