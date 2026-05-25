A proposta para este trabalho da disciplina é criar um sistema de busca e classificação de animais de estimação! Para isso vocês vão selecionar descritores de imagens e vão avaliar o desempenho de cada descritor e de diferentes combinações de descritores para estas duas tarefas semânticas.

Base de Dados
Tivemos a submissão de 42 animais de estimação diferentes! Preparei uma versão da base de dados com todas as imagens já re-dimencionadas para 256x256 e com nomes padronizados; vou disponibilizar as duas versões para vocês. Preparei um csv também listando todos os arquivos e nomes de pets (as “classes”). Casos com nomes repetidos adicionei um sobrenome da pessoa discente; casos sem nome do pet dei nome de “de_pessoa” com primeiro nome da pessoa que enviou a foto.

O Plano
Vocês devem seguir os seguintes passos:

Selecionar 5 ou mais descritores de imagens (pelo menos 1 deles precisa ser um que não foi visto em aula)
Extrair descritores para todas as imagens na base de dados
Tarefa de Classificação
Dividir a base de dados em conjuntos separados para treino (80%), validação (10%) e teste (10%).
Selecionar um classificador para fazer uso desses descritores; pode usar implementações do scikit-learn ou outras externas (mas nada que demande uso de GPU para classificação, a ideia é focar a maior parte do custo computacional na extração de características)
Pets com menos de 3 fotos não entram na tarefa de classificação
Tarefa de Busca
Com a mesma coleção de descritores, implementar uma tarefa de busca (com uma imagem de entrada retornar um ranking de imagens da base de dados)
Usar distância euclidiana simples par a par usando diferentes combinações de descritores
Usar o método Bag of Visual Words (BoVW) e comparar histogramas de visual words usando distância euclideana
Faça visualizações do descritor de Bag of Visual Words usando métodos de projeção como uMap ou t-SNE
Relatório
O relatório de vocês deve apresentar os resultados obtidos. Para cada descritor:

Justifique a escolha do descritor (qual a semântica que você está tentando capturar?)
Crie uma hipótese sobre como o descritor funcionou (ou não funcionou) para as duas tarefas
Apresente acurácias e matrizes de confusão para os resultados de classificação
Apresente resultados de busca que destacam características interessantes
Crie hipóteses sobre a distribuição de descritores na visualização criada a partir da BoVW
Entrega
Relatório como descrito
Código desenvolvido