# TCC 001 - FedPPE: treinamento e teste

Este README descreve como reproduzir o treinamento inicial do modelo, preparar
o firmware e executar o fluxo de aprendizado federado entre o PC e os MCUs.
O projeto é um protótipo experimental baseado no exemplo Magic Wand da Silicon
Labs. A aplicação ativa é o fluxo FedPPE descrito aqui; o código original do
Magic Wand ainda existe em `magic_wand.cc`, mas não é a implementação usada
nesse fluxo.

## 1. Visão geral: quem treina o quê?

O treinamento acontece em duas etapas, em lugares diferentes:

1. **Treinamento inicial centralizado no PC:** o script Python treina a rede
   MLP inteira a partir dos CSVs. O resultado fornece o *backbone* e uma
   cabeça inicial.
2. **Treinamento local e federado:** o backbone fica congelado no firmware.
   Cada MCU atualiza localmente apenas a cabeça usando dados do próprio IMU e
   rótulos informados manualmente. O PC recebe as cabeças locais e executa
   FedAvg para produzir a próxima cabeça global.

```text
CSV rotulados
    |
    v
PC: treino inicial do MLP inteiro em Python
    |                         \
    | backbone                 \ cabeça inicial (27 parâmetros)
    v                            v
firmware compilado            PC/coordenador BLE
    |                            |
    |                       envia a cabeça global
    v                            v
encoder congelado          MCUs treinam localmente a cabeça
                                 |
                                 v
                       PC agrega as cabeças com FedAvg
                                 |
                                 v
                    fedppe_global_head.json (próxima rodada)
```

O script de treinamento usa a biblioteca padrão do Python, com operações
numéricas implementadas no próprio código; não depende de TensorFlow, PyTorch
ou scikit-learn. A comunicação BLE usa a biblioteca `bleak`. A comunicação
serial opcional usa `pyserial`.

## 2. Arquitetura da rede

```text
14 características -> 16 neurônios ReLU -> 8 valores embedding ReLU
                   -> 3 saídas softmax
```

- **Entrada:** 14 características estatísticas, calculadas a partir de uma
  janela de 32 leituras do IMU.
- **Backbone/encoder:** camadas `14 -> 16 -> 8`, incluindo os valores de
  normalização da entrada. O firmware contém 404 parâmetros do backbone e não
  os atualiza durante o treino local.
- **Cabeça/classificador:** camada `8 -> 3`, com 24 pesos e 3 biases (27
  parâmetros no total). As três saídas correspondem a `parado`, `carregando`
  e `vestindo`.

O PC envia somente os 27 parâmetros da cabeça aos MCUs. O backbone fica no
firmware e sua impressão digital (*fingerprint*) é comparada com a do modelo
JSON antes do treino. Isso evita treinar uma cabeça usando um encoder diferente
do esperado.

## 3. Como o conjunto de dados é lido

O comando `train-backbone` aceita:

- Um arquivo ZIP, por exemplo `nn_offline_training-main.zip`; ou
- Um diretório já extraído com os CSVs.

Para o EPI `Helmet` (valor padrão), o script procura as classes:

| Rótulo no modelo | Pasta no conjunto de dados | ID enviado ao MCU |
|---|---|---:|
| parado | `still` | 0 |
| carregando | `carrying` | 1 |
| vestindo | `wearing` | 2 |

Cada `output.csv` precisa ter as colunas `x`, `y`, `z`, `x_g`, `y_g` e `z_g`.
São lidas 32 linhas consecutivas por janela, sem sobreposição. Linhas finais
que não completam uma janela são descartadas.

Para cada grupo de 32 leituras, o script calcula sete estatísticas do
acelerômetro e sete do giroscópio: média e desvio-padrão da magnitude, média e
desvio-padrão das diferenças entre leituras consecutivas, e desvio-padrão de
cada um dos três eixos. Os valores são multiplicados por `0.001`, produzindo
as 14 entradas da rede.

No código, o mapeamento e a extração estão em `CLASS_TO_FOLDER`,
`parse_csv_stream`, `sensor_features` e `extract_features` em
[`fedppe_pc.py`](./fedppe_pc.py).

## 4. Treinamento inicial do modelo no PC

### Requisitos

- Python 3 instalado e disponível pelo launcher `py`.
- O arquivo ZIP ou diretório do conjunto de dados com a estrutura descrita
  acima.
- Não é necessário instalar biblioteca de aprendizado de máquina para este
  treinamento.

Abra o PowerShell na pasta deste projeto e execute, ajustando o caminho para o
conjunto de dados:

```powershell
py fedppe_pc.py train-backbone --dataset "C:\dados\nn_offline_training-main.zip" --ppe Helmet --epochs 30 --learning-rate 0.01 --seed 7 --output fedppe_mlp.json
```

Se os CSVs estiverem em um diretório extraído, passe o caminho do diretório em
`--dataset` no lugar do ZIP.

### O que o comando faz

1. Carrega os CSVs e converte as leituras em janelas rotuladas.
2. Separa as janelas: uma em cada cinco vai para validação; as demais vão para
   treinamento.
3. Calcula a média e o desvio-padrão das características usando somente o
   conjunto de treinamento.
4. Inicializa e treina o MLP por 30 épocas. Em cada época, embaralha as janelas
   de treinamento e atualiza os pesos exemplo a exemplo por retropropagação e
   descida do gradiente.
5. Ao final de cada época, calcula e imprime a acurácia no conjunto de
   validação, se houver janelas de validação.
6. Salva `fedppe_mlp.json` com metadados, backbone, cabeça inicial e quantidade
   de janelas usadas em treino e validação.

Os valores acima são os padrões do script: 30 épocas, taxa de aprendizado
`0.01` e semente aleatória `7`. A semente torna a inicialização e o
embaralhamento repetíveis quando o conjunto de dados e a ordem de leitura são
os mesmos.

### O que deve aparecer

Durante o treinamento, uma linha por época, por exemplo:

```text
epoch=1/30 validation_accuracy=...
...
epoch=30/30 validation_accuracy=...
Modelo salvo em fedppe_mlp.json
janelas treino=... validação=...
backbone=404 parâmetros; cabeça inicial=27
```

A acurácia mostrada é a acurácia de validação do treinamento inicial no PC. Ela
não é a acurácia de uma rodada FedAvg nem uma avaliação final em novos
participantes.

## 5. Exportar o backbone e atualizar o firmware

Depois de gerar o modelo, exporte o backbone como cabeçalho C:

```powershell
py fedppe_pc.py export-backbone --model fedppe_mlp.json --output fedppe_backbone.h
```

O cabeçalho contém a normalização, os pesos e biases do encoder e a
fingerprint. O firmware inclui `fedppe_backbone.h`. Portanto, ao trocar o
modelo inicial, confirme que o arquivo exportado está no local usado pelo
projeto, recompile e regrave o firmware no kit.

No Simplicity Studio 6:

1. Abra `aiml_soc_magic_wand_baremetal.slcp`.
2. Se a configuração BLE/GATT mudou, salve e regenere o projeto.
3. Faça o build do firmware e grave-o no MCU conectado.
4. Confirme que o firmware iniciado anuncia `TCC001-BLE`.

O arquivo compilado presente neste workspace é
`cmake_gcc\build\base\tcc_001.hex`; use uma imagem atualizada após o build. Se
o JSON e o firmware tiverem backbones diferentes, o cliente Python interrompe
a conexão FedAvg com erro de incompatibilidade de fingerprint.

## 6. Teste de comunicação BLE por eco

O teste de eco valida comunicação BLE, mas **não treina o modelo**. Instale
`bleak` e execute:

```powershell
py -m pip install bleak
py fedppe_ble_test.py
```

O PC procura o dispositivo `TCC001-BLE`, conecta, envia a mensagem
`ping-tcc001` e verifica se o MCU devolve os mesmos bytes por notificação. O
resultado esperado começa com `PASS`.

É possível selecionar o endereço do MCU e uma mensagem:

```powershell
py fedppe_ble_test.py --address AA:BB:CC:DD:EE:FF --message "teste 123"
```

A mensagem do teste de eco deve ter entre 1 e 20 bytes UTF-8. Um dongle BLE
precisa ser reconhecido pelo Windows como adaptador Bluetooth Low Energy; um
dongle que disponibiliza somente uma porta serial não é necessariamente
compatível com a biblioteca `bleak`.

## 7. Uma rodada FedAvg via BLE

### Com um MCU: teste ponta a ponta

Com `bleak` instalado e o firmware compatível gravado, execute:

```powershell
py fedppe_pc.py ble-fedavg --windows-per-class 10 --model fedppe_mlp.json --state experimento_01.json
```

Sem `--addresses`, o programa procura um MCU chamado `TCC001-BLE`. Quando
solicitado, posicione o MCU no estado indicado e pressione Enter. O programa
coleta as classes em sequência: `parado`, `carregando` e `vestindo`.

Com dez janelas por classe, cada MCU faz 30 atualizações locais da cabeça por
rodada. Cada janela reúne 32 leituras do IMU e representa um exemplo rotulado.
O MCU envia a cabeça local e a contagem de janelas ao PC ao terminar.

Uma rodada com um MCU verifica a comunicação, o treino local e o salvamento do
estado, mas não demonstra a média entre vários participantes: a média de um
único modelo é igual ao próprio modelo.

### Com dois ou mais MCUs: agregação entre participantes

Informe os endereços BLE dos MCUs na mesma execução:

```powershell
py fedppe_pc.py ble-fedavg --addresses AA:BB:CC:DD:EE:FF 11:22:33:44:55:66 --windows-per-class 10 --model fedppe_mlp.json --state experimento_01.json
```

O PC conecta-se aos dispositivos **um de cada vez**. Para cada MCU, envia a
mesma cabeça global inicial daquela rodada, orienta o treino das três classes
e recebe a cabeça local. Depois que todos os clientes terminam, o PC calcula,
para cada parâmetro `j`:

```text
global[j] = soma_k(janelas_k * local_k[j]) / soma_k(janelas_k)
```

Com o mesmo número de janelas por MCU, isso equivale à média simples das
cabeças locais. O código de agregação está na função `fedavg` de
[`fedppe_pc.py`](./fedppe_pc.py).

### O que observar

- `Cliente BLE 1/N`, `Cliente BLE 2/N` etc. mostram a ordem dos MCUs.
- O programa pede que o operador coloque cada MCU no estado correspondente a
  cada classe.
- Após cada classe, o programa informa que o treino local terminou.
- Ao concluir todos os clientes, aparece a mensagem `Rodada BLE FedAvg
  concluída` e o estado global é salvo no arquivo informado por `--state`.
- Com os parâmetros padrão, o arquivo é `fedppe_global_head.json`.

Cada execução do comando é **uma rodada**; o programa não repete rodadas
automaticamente. Se o arquivo passado em `--state` já existir, o próximo
comando carrega a cabeça agregada nele e começa a rodada seguinte a partir
dela. Para um experimento novo, use um novo nome de arquivo de estado. Todos
os MCUs de uma rodada recebem a mesma cabeça global inicial antes do treino
local.

## 8. Comunicação serial alternativa

O modo serial é uma alternativa para interagir com um MCU pela porta VCOM. Ele
usa `pyserial`:

```powershell
py -m pip install pyserial
py fedppe_pc.py serial --port COM12 --model fedppe_mlp.json --state fedppe_global_head.json
```

Troque `COM12` pela porta COM virtual do kit e feche outros programas que
estejam usando essa porta. A taxa padrão é 115200 baud. No prompt interativo:

- `train CLASSE JANELAS`: coleta exemplos de uma classe (`0`, `1` ou `2`);
- `dump`: mostra os parâmetros da cabeça local;
- `infer`: consulta o estado do MCU; esse comando não inicia uma inferência
  sob demanda;
- `quit`: encerra a sessão.

O modo serial é voltado à interação manual com um MCU; use `ble-fedavg` com
dois ou mais endereços para testar a agregação entre participantes BLE.

## 9. Reprodutibilidade e limites do protótipo

- Para reproduzir o treinamento inicial, registre a versão exata dos CSVs, o
  caminho/organização dos arquivos, `--ppe`, `--epochs`,
  `--learning-rate` e `--seed`.
- A validação atual seleciona cada quinta janela da lista carregada. Como as
  janelas podem vir de uma mesma gravação, essa divisão não garante separação
  por pessoa, sessão ou arquivo. Para uma avaliação acadêmica independente,
  considere uma divisão por participante ou gravação e documente essa
  alteração.
- O treinamento BLE depende dos movimentos capturados pelo IMU e dos rótulos
  escolhidos manualmente. Só o repositório de CSVs não reproduz as cabeças
  locais obtidas em uma rodada física.
- O protótipo não faz conexões BLE simultâneas, rotulagem automática por
  sensor Hall, pseudo-rotulagem nem avaliação automática da acurácia após
  FedAvg.
- O backbone é treinado inicialmente no PC, mas fica congelado no MCU. O
  aprendizado local e a atualização federada modificam somente os 27
  parâmetros da cabeça.
