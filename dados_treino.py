"""
Base de dados de treinamento do LLM local.
Importado por main.py. Edite aqui para expandir o conhecimento do modelo.

Estrutura:
    CORPUS_TEXTO  — Fase 1: texto corrido sobre temas variados (aprende linguagem)
    CORPUS_QA     — Fase 2: pares de pergunta e resposta (aprende a responder)
    CORPUS_THINK  — Fase 3: raciocinio em etapas explicitas (aprende a pensar)
"""
import os

# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Texto corrido, monobloco, temas diversificados
# ═══════════════════════════════════════════════════════════════════════════════
CORPUS_TEXTO = """
A inteligência artificial é um campo da ciência da computação que surgiu na década de 1950, quando o matemático britânico Alan Turing propôs a famosa questão: as máquinas podem pensar? Desde então, pesquisadores de todo o mundo têm trabalhado para criar sistemas capazes de realizar tarefas que normalmente exigiriam cognição humana. Hoje, a IA está presente em aplicações que vão desde assistentes virtuais e carros autônomos até diagnósticos médicos e sistemas de recomendação de conteúdo nas redes sociais. A evolução dos modelos de linguagem como o GPT e o BERT transformou a forma como computadores processam e geram texto, tornando a comunicação entre humanos e máquinas cada vez mais natural e precisa.

O aprendizado de máquina é a principal subárea da inteligência artificial moderna. Em vez de programar regras explícitas para cada situação, os modelos de machine learning aprendem padrões diretamente a partir de grandes volumes de dados. Esse processo é possível graças a algoritmos de otimização como o gradiente descendente, que ajustam progressivamente os parâmetros internos do modelo minimizando uma função de perda. A quantidade e a qualidade dos dados de treinamento são fatores determinantes para o desempenho final, e técnicas como data augmentation e regularização ajudam a melhorar a generalização para dados não vistos.

As redes neurais profundas revolucionaram o campo da inteligência artificial na última década. Inspiradas na estrutura do cérebro humano, essas redes são compostas por múltiplas camadas de neurônios artificiais que processam informações de maneira hierárquica. As arquiteturas convolucionais, as recorrentes e as baseadas em atenção como os Transformers cada uma tem um domínio de aplicação específico. O mecanismo de atenção permite que o modelo identifique quais partes da entrada são mais relevantes para cada etapa do processamento, sendo fundamental para tarefas de tradução, resumo e geração de texto.

A álgebra linear é a base matemática de todo o aprendizado de máquina. Vetores representam pontos no espaço multidimensional, matrizes representam transformações lineares e o produto matricial captura como as transformações se compõem. As operações de multiplicação de matrizes são o coração do forward pass em redes neurais, onde cada camada transforma a representação de entrada por meio de uma matriz de pesos seguida de uma função de ativação não linear. A decomposição de valores singulares, o autovalor e os autovetores têm aplicações diretas em técnicas de redução de dimensionalidade como PCA e SVD.

A estatística fornece as fundações para entender incerteza, variabilidade e inferência a partir de dados. A distribuição normal ou gaussiana é onipresente na natureza e nas ciências, descrevendo fenômenos que resultam da soma de muitas influências independentes. O teorema do limite central explica por que a média de amostras grandes tende a seguir uma distribuição normal independentemente da distribuição original dos dados. Testes de hipótese, intervalos de confiança e regressão linear são ferramentas fundamentais para extrair conclusões defensáveis a partir de dados empíricos.

A física clássica de Newton descreve o movimento de objetos em escalas cotidianas com grande precisão. As três leis do movimento estabelecem que um objeto permanece em repouso ou em movimento uniforme a menos que uma força atue sobre ele, que a força é igual à massa vezes a aceleração, e que a toda ação corresponde uma reação igual e oposta. A lei da gravitação universal descreve como dois corpos se atraem com força proporcional ao produto de suas massas e inversamente proporcional ao quadrado da distância entre eles. Essas leis explicam desde a queda de uma maçã até as órbitas dos planetas.

A termodinâmica estuda as transformações de energia e as relações entre calor, trabalho e temperatura. O primeiro princípio estabelece a conservação da energia: ela não pode ser criada nem destruída, apenas transformada. O segundo princípio introduz o conceito de entropia, indicando que processos espontâneos tendem a aumentar a desordem de um sistema isolado. O terceiro princípio afirma que a entropia de um cristal perfeito no zero absoluto é zero. Essas leis determinam os limites de eficiência de motores térmicos como os de combustão interna, turbinas a vapor e refrigeradores.

O eletromagnetismo unifica as forças elétrica e magnética numa teoria elegante desenvolvida por James Clerk Maxwell no século XIX. As equações de Maxwell descrevem como campos elétricos e magnéticos se geram mutuamente e se propagam no espaço na forma de ondas eletromagnéticas que viajam à velocidade da luz. O espectro eletromagnético abrange desde ondas de rádio e micro-ondas até raios X e raios gama, passando pela luz visível. A indução eletromagnética é o princípio por trás dos geradores elétricos, transformadores e motores que formam a base da civilização moderna.

A física quântica revolucionou a compreensão da matéria e da energia no século XX. O princípio da incerteza de Heisenberg estabelece que não é possível medir simultaneamente com precisão a posição e o momento de uma partícula. O fenômeno do entrelaçamento quântico permite que duas partículas, mesmo separadas por grandes distâncias, tenham estados correlacionados instantaneamente. A dualidade onda-partícula demonstra que elétrons e fótons se comportam ora como ondas ora como partículas dependendo do experimento. A computação quântica usa qubits em superposição para processar certos problemas exponencialmente mais rápido que computadores clássicos.

A química orgânica estuda os compostos de carbono e suas transformações. O carbono tem a propriedade única de formar quatro ligações covalentes estáveis com outros átomos, possibilitando cadeias longas, anéis e estruturas tridimensionais complexas. Os hidrocarbonetos são os compostos mais simples, formados apenas por carbono e hidrogênio. Grupos funcionais como hidroxila, carbonila, carboxila e amino conferem propriedades específicas às moléculas orgânicas. A bioquímica integra a química orgânica com a biologia para estudar as reações que ocorrem nos seres vivos, como a glicólise, o ciclo de Krebs e a cadeia respiratória mitocondrial.

A genética é o ramo da biologia que estuda a hereditariedade e a variação dos organismos. Os genes são sequências de DNA que codificam proteínas ou regulam a expressão de outros genes. O genoma humano contém aproximadamente 3 bilhões de pares de bases e cerca de 20 mil genes codificantes de proteínas. A epigenética estuda como fatores ambientais podem modificar a expressão gênica sem alterar a sequência do DNA, por meio de mecanismos como metilação do DNA e modificação de histonas. A edição de genoma com CRISPR-Cas9 está revolucionando a medicina, a agricultura e a biotecnologia ao permitir modificar sequências específicas de DNA com precisão sem precedentes.

A teoria celular estabelece que todos os seres vivos são compostos por células, que a célula é a unidade básica de vida e que novas células se originam de células preexistentes. As células procarióticas, como as bactérias, não possuem núcleo definido por membrana. As células eucarióticas, presentes em animais, plantas e fungos, possuem núcleo e organelas membranosas especializadas como mitocôndrias, cloroplastos e retículo endoplasmático. A divisão celular por mitose produz células geneticamente idênticas para crescimento e regeneração, enquanto a meiose produz gametas com metade do número de cromossomos para a reprodução sexuada.

A história da humanidade começa com o surgimento do Homo sapiens na África há cerca de 300 mil anos. As primeiras civilizações surgem no Crescente Fértil por volta de 4000 a.C., com as cidades-Estado sumérias desenvolvendo escrita cuneiforme, leis e comércio organizado. O Egito antigo construiu uma das mais duradouras civilizações da história, com faraós, pirâmides e uma rica cultura religiosa e artística. A Grécia clássica entre os séculos V e IV a.C. produziu avanços extraordinários em filosofia, matemática, teatro, escultura e política democrática que influenciam o Ocidente até hoje.

O Império Romano começou como uma pequena cidade-Estado na Península Itálica e expandiu-se até dominar todo o mundo mediterrâneo. Os romanos contribuíram com o direito codificado, a engenharia civil com estradas e aquedutos, o latim que deu origem às línguas românicas e o modelo de administração estatal. A queda do Império Romano do Ocidente em 476 d.C. marcou o início da Idade Média, período de reorganização política, ascensão do feudalismo e expansão do Cristianismo na Europa. As Cruzadas dos séculos XI ao XIII, a Peste Negra do século XIV e a Reforma Protestante do século XVI foram eventos que moldaram profundamente a Europa medieval e moderna.

O Renascimento europeu dos séculos XIV ao XVII representou uma redescoberta dos valores e saberes da Antiguidade Clássica, combinada com uma valorização do humano e do mundo natural. Artistas como Leonardo da Vinci, Miguel Ângelo e Rafael criaram obras de beleza e técnica sem precedentes. Cientistas como Copérnico, Galileu e Kepler desafiaram a visão geocêntrica do universo, inaugurando a Revolução Científica. A invenção da imprensa por Gutenberg por volta de 1450 democratizou o acesso ao conhecimento e acelerou a difusão das novas ideias pela Europa.

A Revolução Francesa de 1789 foi um marco na história política mundial. Impulsionada pelos ideais iluministas de liberdade, igualdade e fraternidade, derrubou o Antigo Regime monárquico e estabeleceu princípios republicanos. Napoleão Bonaparte emergiu das guerras revolucionárias para dominar a Europa, espalhando os códigos jurídicos franceses e os ideais da Revolução. O século XIX viu a consolidação dos estados nacionais, o imperialismo europeu na África e na Ásia, e a crescente tensão entre potências que culminaria nas duas guerras mundiais do século XX.

O Brasil pré-colombiano era habitado por centenas de povos indígenas com culturas, línguas e modos de vida distintos. A chegada dos portugueses em 1500 iniciou um processo de colonização marcado pela exploração de recursos naturais, pelo tráfico transatlântico de africanos escravizados e pela catequização dos povos indígenas pelos jesuítas. A transferência da família real portuguesa para o Brasil em 1808 acelerou o processo de emancipação, culminando na Independência em 1822 com Dom Pedro I. O período imperial durou até a Proclamação da República em 1889, seguida de uma história de alternâncias entre democracia e regimes autoritários ao longo do século XX.

A filosofia grega buscava respostas racionais para as grandes questões da existência. Os pré-socráticos investigaram a natureza fundamental da realidade. Sócrates direcionou a filosofia para questões éticas usando o método da maiêutica. Platão desenvolveu a teoria das Formas e fundou a Academia em Atenas. Aristóteles sistematizou a lógica, fundou a biologia como ciência e desenvolveu a filosofia da mente, da ética e da política. Na modernidade, Descartes inaugurou o racionalismo com o cogito ergo sum, Hume fundou o empirismo britânico e Kant tentou reconciliar as duas tradições na sua filosofia crítica.

A ética é o ramo da filosofia que investiga o que é bom, correto e justo. O utilitarismo de Bentham e Mill propõe que a ação correta maximiza o bem-estar geral. A deontologia kantiana argumenta que certas ações são intrinsecamente certas ou erradas independentemente de suas consequências, baseando-se no imperativo categórico. A ética das virtudes de Aristóteles foca no caráter e nas disposições do agente em vez de regras ou consequências. A ética do cuidado, desenvolvida por Nel Noddings e Carol Gilligan, enfatiza relações concretas e responsabilidade pelo bem-estar dos outros como fundamento moral.

O sistema cardiovascular é fundamental para a manutenção da vida. O coração, um músculo oco com quatro câmaras, bombeia cerca de 5 litros de sangue por minuto em repouso, executando cerca de 100 mil batimentos por dia. As artérias levam sangue oxigenado do coração para os tecidos e as veias retornam o sangue desoxigenado ao coração. Os capilares, vasos microscópicos com paredes de apenas uma célula de espessura, permitem as trocas de gases, nutrientes e resíduos metabólicos entre o sangue e as células. Doenças cardiovasculares como infarto do miocárdio, acidente vascular cerebral e hipertensão são as principais causas de morte no mundo.

O sistema nervoso humano coordena todas as funções do organismo. O cérebro contém aproximadamente 86 bilhões de neurônios conectados por trilhões de sinapses. O córtex cerebral é dividido em lobos com funções especializadas: frontal para raciocínio e controle motor, parietal para processamento sensorial e espacial, temporal para linguagem e memória, e occipital para visão. O sistema nervoso autônomo regula funções involuntárias como frequência cardíaca, respiração e digestão, dividindo-se em simpático, que mobiliza o organismo para ação, e parassimpático, que promove o descanso e a recuperação.

A endocrinologia estuda as glândulas endócrinas e os hormônios que produzem. A hipófise, na base do cérebro, controla a maioria das outras glândulas endócrinas. A tireoide regula o metabolismo por meio dos hormônios T3 e T4. As suprarrenais produzem cortisol em resposta ao estresse e adrenalina em situações de perigo. O pâncreas produz insulina e glucagon para regular os níveis de glicose no sangue. O eixo hipotálamo-hipófise-adrenal é o principal sistema de resposta ao estresse, com impactos em imunidade, metabolismo, crescimento e reprodução.

A psicologia do desenvolvimento estuda como os seres humanos mudam ao longo da vida. Piaget identificou estágios do desenvolvimento cognitivo. Vygotsky enfatizou o papel da interação social e da linguagem. Erikson propôs oito estágios do desenvolvimento psicossocial. Bowlby desenvolveu a teoria do apego, mostrando que vínculos seguros na infância são fundamentais para o desenvolvimento emocional e social. A adolescência é marcada pela busca de identidade, o desenvolvimento do pensamento abstrato e a influência crescente dos pares em relação aos pais.

Os vieses cognitivos são erros sistemáticos de pensamento. O viés de confirmação leva a buscar informações que confirmam crenças preexistentes. A heurística da disponibilidade leva a superestimar a probabilidade de eventos que vêm facilmente à mente. O efeito de ancoragem faz com que estimativas sejam influenciadas por números apresentados anteriormente. O efeito Dunning-Kruger indica que pessoas com pouco conhecimento tendem a superestimar sua competência. A falácia do custo irrecuperável leva a continuar investindo em projetos ruins por causa do que já foi gasto. Reconhecer esses vieses é fundamental para o pensamento crítico e a tomada de decisão racional.

A macroeconomia analisa fenômenos econômicos em escala agregada. O Produto Interno Bruto mede o valor total de bens e serviços produzidos num país num período. A curva de Phillips descreve a relação inversa entre inflação e desemprego no curto prazo. O modelo IS-LM representa o equilíbrio simultâneo nos mercados de bens e de dinheiro. A teoria monetária de Friedman argumenta que inflação é sempre e em todo lugar um fenômeno monetário. A teoria keynesiana defende o papel ativo do governo na estabilização econômica por meio de políticas fiscal e monetária anticíclicas.

O mercado financeiro permite a compra e venda de ativos como ações, títulos, moedas e derivativos. As bolsas de valores são mercados organizados onde empresas captam recursos emitindo ações. O preço de uma ação reflete as expectativas sobre os lucros futuros da empresa. A hipótese dos mercados eficientes de Eugene Fama argumenta que os preços de mercado refletem toda a informação disponível, tornando impossível superar o mercado consistentemente. A análise fundamentalista avalia o valor intrínseco das empresas examinando demonstrações financeiras, gestão e perspectivas do setor.

A ecologia de populações estuda como o tamanho e a composição das populações mudam ao longo do tempo. O modelo logístico descreve como uma população cresce rapidamente quando pequena mas desacelera conforme se aproxima da capacidade de suporte do ambiente. Predação, parasitismo, competição e mutualismo são as principais interações entre populações. A teoria da biogeografia de ilhas de MacArthur e Wilson demonstra como o equilíbrio entre extinções locais e colonizações determina a biodiversidade em habitats isolados. A conservação biológica aplica esses princípios para proteger espécies ameaçadas e restaurar ecossistemas degradados.

As mudanças climáticas representam o maior desafio ambiental do século XXI. Os gases de efeito estufa, principalmente dióxido de carbono de combustíveis fósseis e metano da agropecuária, intensificam o efeito estufa natural aquecendo a Terra. As consequências incluem elevação do nível do mar, eventos climáticos extremos mais frequentes, deslocamento de populações e extinção de espécies. Os pontos de inflexão ou tipping points como o derretimento das calotas polares, o desaparecimento de florestas tropicais e o degelo do permafrost podem desencadear ciclos de retroalimentação positiva que aceleram o aquecimento globalmente.

A astronomia moderna revelou um universo de escala inimaginável. O sistema solar tem cerca de 4,6 bilhões de anos e se formou a partir de uma nuvem de gás e poeira que colapsou por gravidade. Estrelas se formam em nebulosas por condensação gravitacional, vivem queimando hidrogênio em hélio por fusão nuclear e terminam suas vidas como anãs brancas, estrelas de nêutrons ou buracos negros dependendo de sua massa. As ondas gravitacionais detectadas pelo LIGO em 2015 confirmaram a previsão de Einstein sobre perturbações no espaço-tempo causadas por eventos astronômicos violentos como a fusão de buracos negros.

A microbiologia estuda os micro-organismos invisíveis a olho nu. As bactérias são organismos unicelulares procariontes que habitam praticamente todos os ambientes da Terra. O microbioma intestinal humano contém trilhões de bactérias de centenas de espécies que influenciam a digestão, o sistema imunológico, o humor e até o risco de doenças crônicas. Os vírus são entidades não celulares que sequestram a maquinaria de células hospedeiras para se replicar. Os fungos desempenham papel crucial na decomposição de matéria orgânica e no ciclo de nutrientes, além de serem fontes de antibióticos como a penicilina.

A sociologia estuda a sociedade humana, suas estruturas e dinâmicas. Durkheim estudou a coesão social e os fatos sociais como realidades objetivas externas ao indivíduo. Weber analisou a racionalização da sociedade moderna e a burocracia. Marx desenvolveu a teoria do materialismo histórico, argumentando que as relações de produção determinam as estruturas políticas e culturais. A sociologia contemporânea investiga temas como desigualdade social, mobilidade de classe, gênero, raça, religião, mídia e movimentos sociais numa sociedade cada vez mais globalizada e digitalizada.

A linguagem é uma das capacidades mais sofisticadas do cérebro humano. A área de Broca, no córtex frontal esquerdo, é responsável pela produção da fala. A área de Wernicke, no córtex temporal esquerdo, processa a compreensão da linguagem. A aquisição da linguagem ocorre naturalmente nas crianças durante os primeiros anos de vida, com períodos sensíveis para aprender línguas sem sotaque. A linguagem não é apenas um meio de comunicação mas também um instrumento de pensamento: o vocabulário disponível influencia os conceitos que conseguimos formar e os raciocínios que conseguimos fazer.

A filosofia da ciência investiga os fundamentos, métodos e implicações do conhecimento científico. Karl Popper propôs o falsificacionismo como critério de demarcação: uma teoria é científica se pode ser falsificada por experimentos. Thomas Kuhn argumentou que a ciência progride por revoluções de paradigmas em vez de acumulação linear de conhecimento. A crise de replicabilidade em psicologia e medicina levou a reformas nas práticas de pesquisa, incluindo pré-registro de hipóteses, compartilhamento de dados e ênfase em efeitos replicáveis em vez de resultados estatisticamente significativos isolados.

A música é uma arte universal presente em todas as culturas humanas. A teoria musical ocidental baseia-se em escalas, harmonias e ritmos organizados em sistemas como o temperamento igual que divide a oitava em doze semitons iguais. A música clássica europeia desenvolveu formas complexas como a sinfonia, a sonata, a ópera e o quarteto de cordas. O jazz americano surgiu da fusão de tradições africanas e europeias e influenciou praticamente todos os gêneros populares do século XX. A música popular brasileira é uma das mais ricas do mundo, com o samba, a bossa nova, o forró, o axé, o pagode, o sertanejo universitário e o funk como expressões de diversidade cultural.

Os algoritmos são procedimentos computacionais passo a passo para resolver problemas. A análise de complexidade estuda quantos recursos, tempo e memória, um algoritmo requer em função do tamanho da entrada. Algoritmos de ordenação como quicksort e mergesort têm complexidade O de n log n no caso médio. Estruturas de dados como listas, pilhas, filas, árvores binárias, heaps e grafos organizam informações de formas que facilitam operações específicas. A programação dinâmica resolve problemas complexos dividindo-os em subproblemas sobrepostos e armazenando soluções parciais para evitar recálculos, sendo fundamental em bioinformática, otimização e reconhecimento de padrões.

As redes de computadores conectam dispositivos para compartilhamento de recursos e comunicação. A arquitetura TCP/IP organiza a comunicação em camadas: física, enlace, rede, transporte e aplicação. O protocolo HTTP é a base da World Wide Web, permitindo a transferência de hipertexto entre navegadores e servidores. A criptografia protege a confidencialidade e integridade das comunicações digitais usando algoritmos como RSA, AES e curvas elípticas. As redes definidas por software e a computação em borda estão transformando a arquitetura de redes para suportar a crescente demanda por conectividade de dispositivos IoT e aplicações em tempo real.

A administração é a ciência e a arte de coordenar recursos humanos, financeiros e materiais para atingir objetivos organizacionais. A administração científica de Taylor, a teoria clássica de Fayol, as relações humanas de Mayo, a teoria sistêmica e a administração por objetivos de Drucker formam a evolução histórica do pensamento administrativo. A gestão contemporânea integra abordagens ágeis com scrum e kanban, gestão de processos de negócio, análise de dados para tomada de decisão, cultura organizacional inclusiva e responsabilidade social corporativa como dimensões indissociáveis do desempenho organizacional.

A saúde mental é um estado de bem-estar em que o indivíduo consegue lidar com as tensões normais da vida, trabalhar de forma produtiva e contribuir para a comunidade. A depressão é a principal causa de incapacidade no mundo, afetando mais de 280 milhões de pessoas. A ansiedade, o transtorno bipolar, a esquizofrenia e o transtorno de déficit de atenção e hiperatividade são outras condições prevalentes. A terapia cognitivo-comportamental é considerada o tratamento psicológico mais eficaz e com maior base de evidências para a maioria dos transtornos mentais. Hábitos saudáveis como exercício regular, sono adequado, alimentação equilibrada, conexões sociais e práticas meditativas têm efeitos comprovados na prevenção e tratamento de transtornos mentais.

A arquitetura é a arte e a técnica de projetar e construir espaços habitáveis. Do neoclassicismo ao modernismo, a arquitetura reflete os valores, a tecnologia e as aspirações de cada época. O movimento modernista do século XX, com figuras como Le Corbusier, Mies van der Rohe e Frank Lloyd Wright, propôs uma arquitetura funcional, sem ornamentos desnecessários e em harmonia com os materiais modernos como concreto, aço e vidro. A arquitetura paramétrica contemporânea usa algoritmos computacionais para gerar formas complexas e otimizar estruturas. A arquitetura sustentável busca minimizar o impacto ambiental dos edifícios por meio de eficiência energética, materiais ecológicos e integração com a natureza.

O esporte é uma atividade física organizada por regras com funções físicas, psicológicas e sociais. O futebol é o esporte mais popular do mundo com mais de 4 bilhões de seguidores. As Olimpíadas, realizadas a cada quatro anos, reúnem atletas de praticamente todos os países em celebração do esporte e da paz. A psicologia esportiva estuda fatores como motivação, concentração, controle emocional e trabalho em equipe que influenciam o desempenho atlético. O exercício regular reduz o risco de doenças cardiovasculares, diabetes, depressão e demência em todas as idades, sendo considerado um dos hábitos de vida mais poderosos para a saúde.

A nutrição é a ciência que estuda a relação entre alimentos e saúde. Os macronutrientes carboidratos, proteínas e gorduras fornecem energia e são necessários em grandes quantidades. Os micronutrientes vitaminas e minerais são necessários em pequenas quantidades mas essenciais para processos metabólicos específicos. O intestino é considerado o segundo cérebro, com mais de 100 milhões de neurônios entéricos e uma comunicação bidirecional com o cérebro pelo nervo vago, o eixo intestino-cérebro. O padrão alimentar mediterrâneo, rico em vegetais, azeite, leguminosas, nozes e peixe, é consistentemente associado a menor risco de doenças crônicas e longevidade.

A educação formal estrutura o desenvolvimento humano em fases: educação infantil que estimula o desenvolvimento sensório-motor e social, ensino fundamental que sistematiza o letramento e o numeramento, ensino médio que aprofunda o conhecimento e prepara para a vida adulta, e ensino superior que especializa e produz novo conhecimento. Além da educação formal, a educação informal ocorre em todos os ambientes de vida. A tecnologia digital está transformando a educação por meio de ambientes virtuais de aprendizagem, cursos online massivos abertos, gamificação e uso de inteligência artificial para personalização do ensino.

O direito é o conjunto de normas e princípios que regulam a vida em sociedade. O sistema jurídico brasileiro é baseado no direito romano-germânico e tem como sua principal lei a Constituição Federal de 1988. O direito se divide em ramos como civil, penal, trabalhista, administrativo, tributário, ambiental, digital e constitucional. O direito processual define as regras do jogo nos litígios: como propor ações, como produzir provas, como recorrer das decisões. O princípio da presunção de inocência garante que ninguém seja considerado culpado antes de sentença condenatória transitada em julgado.

A política estuda o exercício e distribuição do poder nas sociedades. Os sistemas políticos variam de democracias liberais a autocracias, passando por regimes híbridos. Nas democracias, a legitimidade do poder deriva do consentimento dos governados expresso em eleições periódicas, livres e justas. Os partidos políticos agregam interesses e propõem programas de governo. As instituições como parlamentos, judiciários independentes e imprensa livre são fundamentais para o equilíbrio de poderes e a prevenção de abusos. A participação cidadã por meio de voto, engajamento em movimentos sociais e controle social da administração pública é essencial para a saúde democrática.

A astronomia e a cosmologia revelam a estrutura em grande escala do universo. Aglomerados de galáxias, os maiores objetos ligados gravitacionalmente no universo, estão conectados por filamentos de matéria formando uma teia cósmica com grandes vazios entre eles. A matéria escura, que não emite nem absorve luz mas interage gravitacionalmente, compõe cerca de 27% da densidade total do universo. A energia escura, a mais misteriosa componente do cosmos, compõe cerca de 68% do universo e é responsável pela expansão acelerada do espaço. Somente cerca de 5% do universo é matéria comum que podemos observar diretamente.

A engenharia genética e a biotecnologia estão transformando a medicina, a agricultura e a indústria. A tecnologia de edição genômica CRISPR-Cas9 permite modificar genes com precisão, oferecendo possíveis tratamentos para doenças genéticas como a anemia falciforme, a distrofia muscular de Duchenne e certas formas de câncer. As terapias gênicas buscam corrigir defeitos genéticos em células somáticas ou da linhagem germinativa. Os organismos geneticamente modificados, ou OGMs, são usados na agricultura para aumentar a resistência a pragas, a tolerância à seca e o valor nutricional de culturas. A biologia sintética vai além da modificação de organismos existentes para criar organismos inteiramente novos com funções programadas.

A robótica é a área que projeta, constrói e programa robôs para executar tarefas automaticamente. A robótica industrial revolucionou as linhas de montagem de automóveis, eletrônicos e outros bens manufaturados. Robôs cirúrgicos como o Da Vinci permitem operações minimamente invasivas com maior precisão e menor trauma. Veículos autônomos usam câmeras, lidar, radar e IA para navegar no tráfego sem intervenção humana. Robôs de serviço assistem idosos, entregam mercadorias, exploram ambientes perigosos e realizam tarefas de limpeza e manutenção. Exoesqueletos robóticos estão sendo desenvolvidos para reabilitar pacientes com lesões medulares e ampliar capacidades físicas em trabalhadores industriais.

O empreendedorismo envolve identificar oportunidades, reunir recursos e criar valor por meio de organizações novas ou existentes. As startups são empresas nascentes com modelo de negócios escalável e inovador, frequentemente financiadas por anjos investidores e fundos de capital de risco. O ecossistema empreendedor brasileiro cresceu muito com hubs em São Paulo, Rio de Janeiro, Florianópolis e Belo Horizonte. A metodologia Lean Startup de Eric Ries propõe ciclos rápidos de aprendizado com produtos mínimos viáveis para validar hipóteses de negócio antes de grandes investimentos. O design thinking é uma abordagem centrada no usuário para a resolução criativa de problemas complexos em qualquer domínio.

A comunicação é o processo de transmissão de informações, ideias e emoções entre seres. A comunicação verbal usa palavras e linguagem, enquanto a comunicação não verbal usa gestos, expressões faciais, postura corporal, tom de voz e proximidade física. A teoria da comunicação de Shannon e Weaver modela a transmissão de informação como codificação de uma mensagem, transmissão por um canal com ruído e decodificação pelo receptor. A comunicação interpessoal eficaz requer escuta ativa, empatia, clareza e capacidade de dar e receber feedback. As redes sociais digitais transformaram radicalmente os padrões de comunicação, criando câmaras de eco, acelerando a propagação de informação e desinformação.

A gastronomia é a arte e a ciência da preparação e apreciação de alimentos. A culinária é uma expressão cultural profunda, refletindo história, geografia, religião e valores sociais de cada povo. A culinária brasileira é diversíssima, resultado da fusão de influências indígenas, africanas e europeias com os ingredientes do território mais biodiverso do planeta. O feijão com arroz, a feijoada, o pão de queijo, o acarajé, o vatapá, o baião de dois e a tapioca são expressões da riqueza culinária regional brasileira. A gastronomia molecular aplica técnicas científicas como gelificação, esferificação e emulsificação para criar experiências culinárias inovadoras.

A neurociência cognitiva investiga como o cérebro produz a mente. A memória se divide em memória de trabalho, de curto prazo e de longo prazo, com a última subdividida em memória declarativa, que inclui memória episódica para eventos e semântica para fatos, e memória procedural para habilidades motoras e hábitos. O hipocampo é crucial para a formação de novas memórias episódicas. A atenção é um processo seletivo que direciona os recursos cognitivos para informações relevantes e filtra as irrelevantes. A consciência é talvez o maior mistério da ciência: como processos físicos no cérebro geram a experiência subjetiva, o que filósofos chamam de qualia.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Pares de Pergunta e Resposta estruturados
# ═══════════════════════════════════════════════════════════════════════════════
CORPUS_QA = """
Pergunta: Oi, tudo bem?
Resposta: Olá! Sim, estou bem, obrigado. Como posso te ajudar hoje?

Pergunta: Oi!
Resposta: Oi! Tudo bem por aí? Em que posso ajudar?

Pergunta: Bom dia!
Resposta: Bom dia! Espero que esteja tendo um ótimo dia. O que posso fazer por você?

Pergunta: Boa tarde!
Resposta: Boa tarde! Como posso ajudar você hoje?

Pergunta: Boa noite!
Resposta: Boa noite! Em que posso ser útil?

Pergunta: Como você está?
Resposta: Estou bem, pronto para ajudar! Qual é a sua dúvida ou assunto?

Pergunta: Quem é você?
Resposta: Sou um assistente de inteligência artificial treinado para responder perguntas e ajudar com informações sobre diversos temas. Como posso ajudar?

Pergunta: O que você sabe fazer?
Resposta: Posso responder perguntas sobre ciência, tecnologia, história, filosofia, medicina, economia e muitos outros temas. Também posso ajudar a explicar conceitos e raciocinar sobre problemas. O que você gostaria de saber?

Pergunta: Me conta uma curiosidade interessante.
Resposta: Sabia que o cérebro humano consome cerca de 20% de toda a energia do corpo, apesar de representar apenas 2% da massa corporal? É o órgão mais eficiente e complexo conhecido na natureza.

Pergunta: Qual é a capital do Brasil?
Resposta: A capital do Brasil é Brasília, inaugurada em 21 de abril de 1960. Antes disso, a capital era o Rio de Janeiro.

Pergunta: Quanto é 2 mais 2?
Resposta: 2 mais 2 é igual a 4.

Pergunta: Você consegue me ajudar?
Resposta: Sim, com prazer! Me diga o que você precisa e farei o meu melhor para ajudar.

Pergunta: Obrigado!
Resposta: De nada! Fico feliz em ajudar. Se tiver mais alguma dúvida, é só perguntar.

Pergunta: Até logo!
Resposta: Até logo! Foi um prazer conversar com você. Volte quando quiser.

Pergunta: O que é inteligência artificial?
Resposta: Inteligência artificial é a área da computação que cria sistemas capazes de realizar tarefas que normalmente exigiriam inteligência humana, como reconhecer padrões, tomar decisões e entender linguagem natural.

Pergunta: Como funciona o aprendizado de máquina?
Resposta: O aprendizado de máquina usa algoritmos para encontrar padrões em dados. O modelo ajusta seus parâmetros internos durante o treinamento para minimizar o erro entre suas previsões e os valores corretos, aprendendo a generalizar para dados novos.

Pergunta: O que é uma rede neural profunda?
Resposta: Uma rede neural profunda é um modelo computacional com múltiplas camadas de processamento interconectadas. Cada camada aprende representações progressivamente mais abstratas dos dados, permitindo reconhecer padrões complexos em imagens, texto e áudio.

Pergunta: O que é o Big Bang?
Resposta: O Big Bang é o modelo cosmológico que descreve a origem do universo há 13,8 bilhões de anos, quando toda a matéria e energia existentes estavam concentradas em um ponto extremamente denso e quente que se expandiu rapidamente, dando origem ao espaço, ao tempo e a toda a matéria.

Pergunta: O que é a teoria da evolução?
Resposta: A teoria da evolução proposta por Darwin explica como as espécies se transformam ao longo do tempo por seleção natural. Organismos com características mais adaptadas ao ambiente sobrevivem e reproduzem mais, transmitindo essas características às gerações seguintes.

Pergunta: O que é o DNA?
Resposta: O DNA é a molécula que armazena as informações genéticas de todos os seres vivos. Composto por duas fitas em dupla hélice formadas por nucleotídeos, contém as instruções para produzir todas as proteínas do organismo e é transmitido de pais para filhos na reprodução.

Pergunta: O que é a Constituição Federal do Brasil?
Resposta: A Constituição Federal do Brasil de 1988, chamada de Constituição Cidadã, é a lei máxima do país. Ela define os direitos e deveres dos cidadãos, organiza os poderes do Estado e estabelece os fundamentos da República como soberania, cidadania e dignidade da pessoa humana.

Pergunta: O que é inflação?
Resposta: Inflação é o aumento generalizado e contínuo dos preços de bens e serviços numa economia, resultando na perda do poder de compra da moeda. É medida por índices como o IPCA no Brasil e controlada principalmente pela política de juros do banco central.

Pergunta: O que é fotossíntese?
Resposta: Fotossíntese é o processo pelo qual plantas, algas e cianobactérias convertem luz solar, dióxido de carbono e água em glicose e oxigênio, utilizando cloroplastos. É a base de quase toda a vida na Terra e a principal fonte do oxigênio atmosférico.

Pergunta: O que é a relatividade especial de Einstein?
Resposta: A relatividade especial, proposta em 1905, estabelece que as leis da física são iguais para todos os observadores em movimento uniforme e que a velocidade da luz é constante para todos. Uma consequência é a equivalência entre massa e energia, expressa por E igual a mc ao quadrado.

Pergunta: O que é globalização?
Resposta: Globalização é o processo de integração econômica, política, cultural e tecnológica entre os países do mundo, intensificado pela revolução dos transportes e das telecomunicações. Ela facilita o comércio internacional, a difusão cultural e o fluxo de capital, mas também pode aprofundar desigualdades entre países e dentro deles.

Pergunta: O que é neuroplasticidade?
Resposta: Neuroplasticidade é a capacidade do cérebro de se reorganizar, formar novas conexões neurais e se adaptar em resposta a experiências, aprendizado e lesões. Essa capacidade é maior na infância mas persiste ao longo de toda a vida, sendo a base do aprendizado e da reabilitação neurológica.

Pergunta: O que é o método científico?
Resposta: O método científico é um processo sistemático para produzir conhecimento confiável. Envolve observação, formulação de hipóteses, experimentos controlados, análise de dados e revisão por pares, permitindo que as conclusões sejam testadas e replicadas independentemente por outros pesquisadores.

Pergunta: O que é sustentabilidade?
Resposta: Sustentabilidade é o princípio de satisfazer as necessidades do presente sem comprometer a capacidade das gerações futuras. Envolve equilíbrio entre desenvolvimento econômico, equidade social e preservação ambiental, integrando as dimensões econômica, social e ecológica das decisões humanas.

Pergunta: O que é democracia?
Resposta: Democracia é um sistema de governo onde o poder emana do povo, que o exerce diretamente ou por meio de representantes eleitos. Seus princípios fundamentais incluem sufrágio universal, separação de poderes, liberdade de expressão, estado de direito e proteção dos direitos das minorias.

Pergunta: O que são direitos humanos?
Resposta: Direitos humanos são direitos fundamentais inerentes a todas as pessoas, independentemente de raça, sexo, nacionalidade, idioma ou religião. Foram formalmente reconhecidos na Declaração Universal dos Direitos Humanos da ONU em 1948 e incluem vida, liberdade, igualdade e dignidade.

Pergunta: O que é a cadeia alimentar?
Resposta: A cadeia alimentar descreve o fluxo de energia e nutrientes entre organismos num ecossistema. Começa com produtores como plantas que convertem energia solar. Passa pelos consumidores primários, secundários e terciários, e termina nos decompositores que reciclam nutrientes para o solo.

Pergunta: Como funciona o sistema imunológico?
Resposta: O sistema imunológico defende o organismo contra patógenos. Inclui barreiras físicas como pele, células como leucócitos e linfócitos, e proteínas como anticorpos. Quando reconhece um agente estranho, desencadeia resposta inflamatória, produz anticorpos específicos e cria células de memória para respostas futuras mais rápidas.

Pergunta: O que é processamento de linguagem natural?
Resposta: Processamento de linguagem natural, o NLP, é a área da IA que permite que computadores entendam, interpretem e gerem linguagem humana. É usado em tradução automática, chatbots, análise de sentimento, resumo automático de textos e assistentes virtuais como Siri e Alexa.

Pergunta: O que é energia renovável?
Resposta: Energia renovável é obtida de fontes naturais que se renovam continuamente, como solar, eólica, hidroelétrica, geotérmica e biomassa. Ao contrário dos combustíveis fósseis, emitem pouco ou nenhum carbono e não se esgotam em escala humana de tempo, sendo fundamentais para a transição energética.

Pergunta: O que é blockchain?
Resposta: Blockchain é uma tecnologia de registro distribuído onde dados são armazenados em blocos encadeados criptograficamente, tornando-os praticamente imutáveis e auditáveis. Cada bloco contém transações e o hash do bloco anterior, criando uma cadeia resistente a fraudes sem necessidade de autoridade central.

Pergunta: O que é o efeito estufa?
Resposta: O efeito estufa é o fenômeno pelo qual gases como dióxido de carbono, metano e vapor de água retêm parte do calor solar na atmosfera terrestre, mantendo a temperatura adequada para a vida. O aumento desses gases pela ação humana está intensificando esse efeito e causando o aquecimento global.

Pergunta: O que é inteligência emocional?
Resposta: Inteligência emocional é a capacidade de identificar, compreender, gerenciar e usar eficazmente emoções próprias e dos outros. Popularizada por Daniel Goleman, inclui autoconsciência, autorregulação, motivação intrínseca, empatia e habilidades sociais como comunicação eficaz e resolução de conflitos.

Pergunta: O que é a segunda lei da termodinâmica?
Resposta: A segunda lei da termodinâmica establece que a entropia de um sistema isolado tende a aumentar ao longo do tempo. Isso significa que processos naturais são irreversíveis, que não existe motor térmico 100% eficiente e que o calor flui espontaneamente do corpo mais quente para o mais frio.

Pergunta: O que é o princípio da incerteza de Heisenberg?
Resposta: O princípio da incerteza de Heisenberg establece que é impossível conhecer simultaneamente com precisão arbitrária a posição e o momento de uma partícula quântica. Quanto mais precisamente medimos uma dessas grandezas, menos precisamente podemos conhecer a outra, o que é uma característica fundamental da natureza e não uma limitação tecnológica.

Pergunta: O que é CRISPR?
Resposta: CRISPR-Cas9 é uma tecnologia de edição genômica que permite modificar sequências específicas de DNA com alta precisão e baixo custo. Derivada de um mecanismo de defesa de bactérias contra vírus, ela usa uma molécula de RNA guia para localizar o gene alvo e uma proteína Cas9 para cortar o DNA nesse ponto.

Pergunta: O que é o PIB?
Resposta: O Produto Interno Bruto, PIB, mede o valor total de bens e serviços finais produzidos numa economia num período de tempo. É o principal indicador do tamanho e do crescimento econômico de um país, embora seja criticado por não capturar desigualdade de renda, qualidade de vida ou sustentabilidade ambiental.

Pergunta: O que é a teoria das cordas?
Resposta: A teoria das cordas é uma abordagem teórica que propõe que as partículas fundamentais não são pontuais mas sim cordas uni dimensionais vibrando em múltiplas dimensões do espaço-tempo. Busca unificar a mecânica quântica com a relatividade geral, mas ainda carece de confirmações experimentais diretas.

Pergunta: O que é machine learning não supervisionado?
Resposta: No machine learning não supervisionado, o modelo encontra padrões em dados sem rótulos, sem uma resposta correta pré-definida. É usado em agrupamento de dados, detecção de anomalias, redução de dimensionalidade e descoberta de estruturas ocultas, como segmentação de clientes e detecção de fraudes.

Pergunta: O que é epigenética?
Resposta: Epigenética estuda como fatores ambientais modificam a expressão gênica sem alterar a sequência do DNA. Mecanismos como metilação do DNA e modificação de histonas controlam quais genes são ativados ou silenciados. Essas modificações podem ser influenciadas pela dieta, estresse, exposição a toxinas e até serem transmitidas para as gerações seguintes.

Pergunta: O que é a teoria do apego?
Resposta: A teoria do apego, desenvolvida por John Bowlby e Mary Ainsworth, descreve como vínculos emocionais entre crianças e seus cuidadores influenciam o desenvolvimento psicológico. Apego seguro, ansioso, evitativo e desorganizado são padrões identificados com impactos duradouros na saúde mental, relacionamentos e capacidade de regular emoções na vida adulta.

Pergunta: O que é a hipótese de Gaia?
Resposta: A hipótese de Gaia, proposta por James Lovelock, sugere que a Terra funciona como um sistema autorregulado onde os seres vivos interagem com o ambiente inorgânico de forma a manter condições favoráveis à vida. Embora controversa em sua versão mais forte, a ideia de que os seres vivos influenciam ativamente os ciclos biogeoquímicos da Terra é amplamente aceita na ciência.

Pergunta: O que é processamento paralelo em computação?
Resposta: Processamento paralelo é a execução simultânea de múltiplas tarefas computacionais em diferentes processadores ou núcleos. Em redes neurais, a computação paralela em GPUs com milhares de núcleos acelera dramaticamente o treinamento de modelos ao processar muitos exemplos simultaneamente. É fundamental para o desenvolvimento moderno de IA.

Pergunta: O que é o microbioma humano?
Resposta: O microbioma humano é o conjunto de trilhões de micro-organismos como bactérias, fungos, vírus e arqueas que habitam o corpo humano, especialmente o intestino. Influencia a digestão, o sistema imunológico, a produção de vitaminas, a regulação hormonal e até o humor e o comportamento por meio do eixo intestino-cérebro.

Pergunta: O que é uma estrela de nêutrons?
Resposta: Uma estrela de nêutrons é o remanescente compacto de uma estrela massiva após uma supernova. Com massa de 1 a 3 vezes a do Sol comprimida numa esfera de cerca de 20 km de diâmetro, tem densidade extrema onde prótons e elétrons se fundem em nêutrons. Pulsares são estrelas de nêutrons que emitem jatos de ondas de rádio enquanto giram rapidamente.

Pergunta: O que é o viés de confirmação?
Resposta: O viés de confirmação é a tendência cognitiva de buscar, interpretar, favorecer e recordar informações de um modo que confirme nossas crenças ou hipóteses preexistentes, ignorando ou minimizando evidências contrárias. É um dos vieses cognitivos mais poderosos e pode prejudicar a tomada de decisão em todas as áreas da vida.

Pergunta: O que é direito penal?
Resposta: Direito penal é o ramo do direito que define quais condutas são consideradas crimes ou contravenções e estabelece as sanções correspondentes. Baseia-se em princípios como legalidade, que proíbe crime sem lei prévia, culpabilidade, que exige responsabilidade subjetiva, e proporcionalidade, que exige penas compatíveis com a gravidade da conduta.

Pergunta: O que é o ciclo de Krebs?
Resposta: O ciclo de Krebs, ou ciclo do ácido cítrico, é uma série de reações químicas que ocorrem na mitocôndria e são fundamentais para o metabolismo aeróbico. Usando o acetil-CoA derivado da glicose, gorduras e proteínas, produz o NADH e o FADH2 que alimentam a cadeia respiratória para a síntese de ATP, a moeda energética celular.

Pergunta: O que é consciência?
Resposta: A consciência é a experiência subjetiva de ser e de perceber, o aspecto de como é ser uma pessoa tendo pensamentos, sensações e emoções. É considerada um dos maiores mistérios da ciência e da filosofia: como processos físicos no cérebro geram experiência subjetiva é o que David Chalmers chamou de o problema difícil da consciência.

Pergunta: O que é deflação?
Resposta: Deflação é a queda generalizada e contínua dos preços de bens e serviços numa economia. Embora pareça benéfica, pode ser perigosa economicamente: consumidores adiam compras esperando preços ainda menores, empresas reduzem produção e investimento, salários caem e o desemprego aumenta, podendo gerar espirais deflacionárias como a da Grande Depressão.

Pergunta: O que é uma black hole?
Resposta: Um buraco negro é uma região do espaço-tempo onde a gravidade é tão intensa que nem mesmo a luz consegue escapar. Forma-se quando uma estrela massiva colapsa sobre si mesma. O horizonte de eventos é o limite de não retorno. No centro existe a singularidade, onde as equações da física conhecida deixam de funcionar.

Pergunta: O que é o teste de Turing?
Resposta: O teste de Turing, proposto por Alan Turing em 1950, é um critério de inteligência para máquinas. Um humano se comunica por escrito tanto com outro humano quanto com uma máquina. Se o avaliador não conseguir distinguir qual é qual, considera-se que a máquina demonstrou comportamento inteligente equivalente ao humano.

Pergunta: O que é clorofila?
Resposta: Clorofila é o pigmento verde presente nos cloroplastos das plantas e algas que absorve luz solar, principalmente nos espectros vermelho e azul, refletindo o verde. É a molécula central da fotossíntese, pois captura a energia luminosa que é convertida em energia química pela planta para produzir glicose a partir de dióxido de carbono e água.

Pergunta: O que é o sistema endócrino?
Resposta: O sistema endócrino é composto por glândulas que produzem e secretam hormônios diretamente na corrente sanguínea, regulando funções como metabolismo, crescimento, reprodução, humor e resposta ao estresse. As principais glândulas endócrinas são hipófise, tireoide, paratireoide, suprarrenais, pâncreas e gônadas.

Pergunta: O que é filosofia analítica?
Resposta: Filosofia analítica é a tradição filosófica dominante no mundo anglofônico, caracterizada pela clareza conceitual, rigor lógico e uso de métodos da lógica e da linguística para abordar problemas filosóficos. Desenvolvida por Frege, Russell, Wittgenstein e Moore no início do século XX, abrange lógica, filosofia da linguagem, epistemologia, filosofia da mente e metafísica.

Pergunta: O que é fermentação?
Resposta: Fermentação é um processo metabólico anaeróbico no qual microrganismos como bactérias e leveduras convertem açúcares em produtos como álcool etílico, dióxido de carbono ou ácido lático, sem usar oxigênio. É usado na produção de pão, cerveja, vinho, queijo, iogurte, vinagre e kimchi, e na produção de biocombustíveis.

Pergunta: O que é inteligência artificial generativa?
Resposta: IA generativa é um tipo de inteligência artificial capaz de criar novos conteúdos como texto, imagens, áudio e código que não existiam anteriormente nos dados de treinamento. Modelos como GPT para texto e Stable Diffusion para imagens aprendem distribuições de dados e geram novos exemplos que parecem ter origem humana.

Pergunta: O que é o método socrático?
Resposta: O método socrático é uma forma de diálogo filosófico onde o interlocutor faz perguntas em sequência para ajudar o outro a examinar suas crenças, identificar contradições e chegar a uma compreensão mais profunda por si mesmo. Também chamado de maiêutica ou dialética, é a base do ensino por questionamento e do pensamento crítico.

Pergunta: O que são neurônios espelho?
Resposta: Neurônios espelho são células nervosas que se ativam tanto quando um indivíduo realiza uma ação quanto quando observa outra pessoa realizando a mesma ação. Descobertos em macacos e identificados em humanos, são considerados o substrato neural da empatia, da imitação e da compreensão das intenções dos outros.

Pergunta: O que é o aquecimento global?
Resposta: Aquecimento global é o aumento da temperatura média da superfície e da atmosfera da Terra, causado principalmente pela emissão de gases de efeito estufa pela atividade humana. A temperatura média já aumentou cerca de 1,2 graus Celsius desde a era pré-industrial, causando derretimento de geleiras, elevação do nível do mar e mais eventos climáticos extremos.

Pergunta: O que é recursão em programação?
Resposta: Recursão é uma técnica de programação em que uma função chama a si mesma para resolver um problema dividindo-o em subproblemas menores e idênticos. Todo algoritmo recursivo precisa de um caso base que interrompe a recursão e um caso recursivo que reduz o problema. Exemplos clássicos incluem cálculo de fatorial, sequência de Fibonacci e busca em árvores.

Pergunta: O que é a lei da oferta e da demanda?
Resposta: A lei da oferta e da demanda descreve como o preço de equilíbrio de um bem é determinado pela interação entre produtores e consumidores. Quando a demanda aumenta ou a oferta diminui, o preço sobe. Quando a demanda diminui ou a oferta aumenta, o preço cai. É o mecanismo fundamental de alocação de recursos em economias de mercado.

Pergunta: O que é a decomposição de resíduos?
Resposta: A decomposição é o processo pelo qual fungos, bactérias e outros decompositores quebram matéria orgânica morta em substâncias mais simples, reciclando nutrientes como carbono, nitrogênio e fósforo de volta ao solo e à atmosfera. É fundamental para o funcionamento dos ecossistemas e o fechamento dos ciclos biogeoquímicos.

Pergunta: O que é aprendizado por reforço?
Resposta: Aprendizado por reforço é um paradigma de machine learning onde um agente aprende a tomar decisões interagindo com um ambiente, recebendo recompensas por ações corretas e penalidades por erros. É a abordagem por trás do AlphaGo, que derrotou o campeão mundial de Go, e de sistemas que aprendem a jogar videogames ou controlar robôs.

Pergunta: O que é a síndrome de burnout?
Resposta: Burnout é um estado de esgotamento físico e emocional causado pelo estresse crônico no trabalho. Caracteriza-se por exaustão, cinismo ou desapego profissional e redução da eficácia. A OMS reconheceu o burnout como um fenômeno ocupacional em 2019. Seu tratamento envolve mudanças no ambiente de trabalho, psicoterapia e, em alguns casos, medicação.

Pergunta: O que é a lei da conservação de energia?
Resposta: A lei da conservação de energia, ou primeiro princípio da termodinâmica, establece que energia não pode ser criada nem destruída, apenas transformada de uma forma em outra. Numa queda d'água, energia potencial gravitacional se converte em cinética e depois em calor e som no impacto. Essa lei é um dos princípios mais fundamentais da física.

Pergunta: O que é o efeito placebo?
Resposta: O efeito placebo é a melhora clínica que ocorre quando um paciente recebe um tratamento inativo, como pílulas de açúcar, acreditando ser um tratamento real. Demonstra o poder da expectativa e da crença sobre processos fisiológicos reais. O efeito nocebo é o oposto: efeitos adversos causados por expectativas negativas sobre um tratamento.

Pergunta: O que é memória de trabalho?
Resposta: Memória de trabalho, ou memória operacional, é um sistema cognitivo que mantém e manipula temporariamente pequenas quantidades de informação para uso imediato. É fundamental para atividades como compreensão de texto, cálculo mental e raciocínio. Tem capacidade limitada, geralmente 7 mais ou menos 2 itens, e dura apenas alguns segundos sem repetição.

Pergunta: O que é a bomba atômica?
Resposta: A bomba atômica é uma arma nuclear que libera energia explosiva a partir de reações de fissão nuclear, onde núcleos pesados como urânio-235 ou plutônio-239 se quebram ao ser atingidos por nêutrons, liberando enorme quantidade de energia e mais nêutrons que sustentam uma reação em cadeia. As bombas de Hiroshima e Nagasaki em 1945 mataram entre 110 e 210 mil pessoas.

Pergunta: O que é a teoria da tectônica de placas?
Resposta: A teoria da tectônica de placas descreve como a litosfera terrestre está dividida em grandes placas que se movem lentamente sobre o manto semi-fluido. O movimento das placas causa terremotos, vulcões, a formação de montanhas e a expansão e retração dos oceanos. É a teoria unificadora da geologia moderna, equivalente à evolução para a biologia.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Raciocínio em etapas explícitas (processo de pensamento profundo)
# ═══════════════════════════════════════════════════════════════════════════════
CORPUS_THINK = """
Pergunta: O que é fotossíntese e por que ela é importante?
[Pensando] Preciso explicar fotossíntese de forma completa. Passo 1: identificar o que é. É um processo biológico. Passo 2: quem realiza? Plantas, algas e cianobactérias, ou seja, organismos autotróficos. Passo 3: o que precisam como entrada? Luz solar como fonte de energia, dióxido de carbono do ar e água do solo. Passo 4: o que produzem como saída? Glicose como combustível celular e oxigênio como subproduto. Passo 5: onde ocorre? Nos cloroplastos, organelas com membranas tilacoides ricas em clorofila. Passo 6: como funciona em detalhes? Fase clara nas membranas tilacoides usa energia luminosa para gerar ATP e NADPH e liberar oxigênio. Fase escura no estroma usa ATP e NADPH para fixar CO2 em glicose pelo ciclo de Calvin. Passo 7: por que é importante? É a base de quase todas as cadeias alimentares da Terra. Passo 8: outro motivo? Produziu e mantém o oxigênio atmosférico que permite a respiração aeróbica. Passo 9: mais um? Captura carbono da atmosfera, sendo fundamental no ciclo do carbono e no controle climático. Passo 10: resumindo o que devo responder com todos esses pontos. [/Pensando]
Resposta: Fotossíntese é o processo pelo qual plantas, algas e cianobactérias convertem luz solar, dióxido de carbono e água em glicose e oxigênio, utilizando os cloroplastos. É fundamental porque é a base de quase todas as cadeias alimentares da Terra, produz o oxigênio atmosférico que permite a vida aeróbica e captura dióxido de carbono da atmosfera, sendo essencial para o equilíbrio climático do planeta.

Pergunta: Por que o céu é azul durante o dia mas fica vermelho ao pôr do sol?
[Pensando] Esta é uma bela questão de óptica atmosférica com duas partes. Passo 1: entender a composição da luz solar. A luz do Sol parece branca mas contém todas as cores do espectro visível, de violeta a vermelho. Passo 2: o que acontece com a luz ao entrar na atmosfera? Ela interage com moléculas de nitrogênio e oxigênio. Passo 3: que processo físico ocorre? Espalhamento de Rayleigh. Passo 4: como funciona esse espalhamento? É inversamente proporcional à quarta potência do comprimento de onda. Isso significa que luz de menor comprimento de onda espalha muito mais. Passo 5: quais cores têm menor comprimento de onda? Violeta e azul. Passo 6: então por que o céu é azul e não violeta se o violeta espalha mais? Porque o Sol emite relativamente menos luz violeta e nossos olhos são muito mais sensíveis ao azul que ao violeta. Passo 7: agora o segundo fenômeno. Por que o céu fica laranja e vermelho ao pôr do sol? Passo 8: raciocínio. Ao entardecer, a luz percorre uma distância muito maior na atmosfera para chegar aos nossos olhos. Passo 9: consequência. Toda a luz de comprimento de onda curto, azul e violeta, já foi espalhada antes de chegar a nós. Passo 10: resta o quê? As cores de comprimento de onda longo como laranja e vermelho, que conseguem atravessar o caminho mais longo sem ser totalmente espalhadas. Passo 11: conclusão integrada sobre os dois fenômenos. [/Pensando]
Resposta: O céu é azul durante o dia porque as moléculas da atmosfera espalham a luz solar de acordo com o espalhamento de Rayleigh, que é muito mais eficiente para comprimentos de onda curtos como o azul. O azul chega aos nossos olhos de todas as direções, tornando o céu azul. No pôr do sol, a luz percorre um caminho muito maior na atmosfera, fazendo com que toda a luz azul já tenha sido espalhada ao longo do trajeto. Chegam aos nossos olhos apenas as cores de comprimento de onda longo, como laranja e vermelho, pintando o céu com essas tonalidades cálidas.

Pergunta: Como funciona a internet e o que é o protocolo TCP/IP?
[Pensando] Passo 1: definir o que é a internet. É uma rede global de redes de computadores interconectadas. Passo 2: como os computadores se comunicam? Por meio de protocolos, conjuntos de regras padronizadas. Passo 3: qual é o protocolo central? TCP/IP, que na verdade são dois protocolos distintos trabalhando juntos. Passo 4: o que é o IP? Internet Protocol, responsável por endereçamento. Cada dispositivo conectado tem um endereço IP único, como um CEP digital. Passo 5: como os dados são enviados? Não como um fluxo contínuo, mas divididos em pacotes menores. Passo 6: o que cada pacote contém? O dado em si, endereço de origem, endereço de destino e informações de sequência. Passo 7: como os pacotes chegam ao destino? Passam por roteadores que decidem o melhor caminho a cada salto, como um sistema postal inteligente. Passo 8: os pacotes chegam em ordem? Não necessariamente. Podem tomar caminhos diferentes e chegar fora de ordem. Passo 9: então quem garante que tudo funcione? O TCP, Transmission Control Protocol. Passo 10: o que o TCP faz? Garante entrega confiável, confirma recebimento de cada pacote, solicita reenvio dos perdidos e reordena os que chegaram fora de sequência. Passo 11: e o DNS? É o sistema de nomes de domínio que traduz nomes legíveis como google.com em endereços IP numéricos. Passo 12: resumo do processo completo para formular a resposta. [/Pensando]
Resposta: A internet é uma rede global de redes interconectadas que se comunicam por meio de protocolos. O IP atribui endereços únicos a cada dispositivo e divide os dados em pacotes que viajam independentemente por roteadores até o destino. O TCP garante a entrega confiável, confirmando o recebimento de cada pacote, solicitando reenvio dos perdidos e reordenando os que chegam fora de sequência. O DNS traduz nomes de domínio como google.com em endereços IP numéricos para que os roteadores saibam para onde enviar os dados.

Pergunta: Por que a água faz tão bem ao organismo humano?
[Pensando] Passo 1: identificar a composição do corpo humano em relação à água. O corpo adulto é composto por cerca de 60% de água. Passo 2: funções estruturais. Água é o solvente universal das reações bioquímicas, dissolvendo nutrientes, gases e resíduos. Passo 3: função de transporte. O sangue é 92% água e transporta oxigênio, hormônios, nutrientes e resíduos por todo o corpo. Passo 4: regulação de temperatura. A evaporação do suor remove calor do corpo, mantendo a temperatura estável. Passo 5: proteção de órgãos e articulações. Líquido sinovial entre articulações, líquido cefalorraquidiano em torno do cérebro e espinha. Passo 6: digestão. Saliva e sucos digestivos são basicamente água com enzimas. Passo 7: excreção. Urina carrega resíduos metabólicos como ureia para fora do corpo. Passo 8: fotossíntese e cadeia alimentar. Toda a comida que consumimos depende da água para existir. Passo 9: impacto da desidratação. Mesmo 2% de desidratação reduz o desempenho cognitivo e físico. Passo 10: desidratação grave. Pode causar confusão mental, falência de órgãos e morte. Passo 11: quanto devo ingerir? A recomendação é cerca de 2 litros por dia para adultos, variando com atividade física, temperatura e saúde. Passo 12: integrar todos os pontos em uma resposta coerente. [/Pensando]
Resposta: A água é essencial para a vida porque desempenha funções vitais no organismo. É o solvente onde ocorrem todas as reações bioquímicas das células. O sangue, composto majoritariamente de água, transporta oxigênio, hormônios e nutrientes para todos os tecidos. A água regula a temperatura corporal pela evaporação do suor, protege articulações e o sistema nervoso, participa da digestão e excreção de resíduos metabólicos. A desidratação de apenas 2% já compromete o desempenho cognitivo e físico, e a perda de 10% ou mais pode ser fatal.

Pergunta: Como funciona uma vacina e por que ela protege?
[Pensando] Passo 1: entender o problema que a vacina resolve. O sistema imunológico precisa aprender a combater um patógeno sem sofrer a doença real. Passo 2: como o sistema imunológico aprende naturalmente? Quando é infectado, produz anticorpos específicos e cria células de memória. Passo 3: o problema do aprendizado natural. A doença real pode ser grave ou letal antes da imunidade ser estabelecida. Passo 4: a solução da vacina. Apresentar ao sistema imunológico o patógeno ou partes dele de forma inofensiva. Passo 5: tipos de vacinas. Vírus inativado ou atenuado como as de poliomielite, fragmento proteico do patógeno como a de hepatite B, RNA mensageiro que instrui células a produzir uma proteína do patógeno como as de Covid-19. Passo 6: o que acontece após a vacinação. O sistema imunológico reconhece o antígeno estranho. Passo 7: resposta imune adaptativa. Linfócitos B produzem anticorpos específicos. Linfócitos T auxiliam e eliminam células infectadas. Passo 8: o que resta após a resposta inicial? Células B e T de memória que persistem por anos ou décadas. Passo 9: como essas células de memória protegem? Na próxima exposição ao patógeno real, respondem rapidamente, em horas em vez de dias, eliminando o patógeno antes que cause doença grave. Passo 10: imunidade de rebanho. Quando uma proporção suficiente da população está imune, o patógeno não encontra hospedeiros suficientes para se propagar, protegendo também os não vacinados. Passo 11: resumo. [/Pensando]
Resposta: Uma vacina ensina o sistema imunológico a reconhecer e combater um patógeno sem causar a doença real. Ela introduz no organismo uma forma inofensiva do agente infeccioso, como vírus inativado, fragmento de proteína ou RNA mensageiro que instrui células a produzir uma proteína do patógeno. O sistema imunológico responde produzindo anticorpos e criando células de memória que persistem por anos. Na exposição futura ao patógeno real, essas células respondem rapidamente, eliminando a infecção antes que cause doença grave. Quando uma proporção suficiente da população está vacinada, estabelece-se a imunidade coletiva que protege até os não vacinados.

Pergunta: O que acontece no cérebro quando aprendemos algo novo?
[Pensando] Passo 1: identificar o fenômeno central. A neuroplasticidade, a capacidade do cérebro de se reorganizar. Passo 2: o que é um neurônio. Célula nervosa que transmite sinais elétricos. Passo 3: o que é uma sinapse. A conexão entre neurônios onde ocorre a transmissão de sinais por neurotransmissores. Passo 4: o que acontece quando aprendemos algo pela primeira vez? Novas conexões sinápticas são formadas entre neurônios. Passo 5: a regra de Hebb. Neurônios que disparam juntos se conectam juntos. A ativação repetida de um circuito neural reforça essa conexão. Passo 6: potenciação de longo prazo. O processo molecular pelo qual sinapses se fortalecem com uso repetido, aumentando a eficiência da transmissão. Passo 7: o papel do sono. Durante o sono, o cérebro consolida memórias, transferindo informações do hipocampo para o córtex cerebral para armazenamento de longo prazo. Passo 8: o papel da repetição. A prática repetida mieliniza os axônios, aumentando a velocidade e a eficiência da transmissão neural. Passo 9: o papel das emoções. O sistema límbico, especialmente a amígdala, marca experiências emocionalmente significativas como mais importantes, facilitando sua retenção. Passo 10: neurogênese. No hipocampo, o cérebro adulto pode gerar novos neurônios, especialmente com exercício físico e aprendizado, embora isso seja mais limitado que na infância. Passo 11: o que enfraquece as memórias. Sinapses pouco usadas se enfraquecem e podem ser eliminadas num processo de poda sináptica que também é fundamental para o aprendizado eficiente. Passo 12: integrar tudo. [/Pensando]
Resposta: Quando aprendemos algo novo, o cérebro forma novas conexões sinápticas entre neurônios, seguindo a regra de que neurônios que disparam juntos se conectam juntos. A repetição fortalece essas sinapses pelo processo de potenciação de longo prazo, aumentando a eficiência da transmissão neural. O sono é fundamental para consolidar memórias, transferindo informações do hipocampo para o córtex para armazenamento duradouro. Emoções fortes marcam experiências como importantes, facilitando sua retenção. A mielinização dos axônios com a prática aumenta a velocidade de processamento, e o uso seletivo de sinapses com poda das não utilizadas torna o aprendizado mais eficiente ao longo do tempo.

Pergunta: Por que a Terra tem estações do ano?
[Pensando] Passo 1: identificar o mito mais comum. Muitas pessoas acham que as estações são causadas pela variação da distância da Terra ao Sol. Passo 2: isso está errado. A Terra está mais próxima do Sol em janeiro, inverno no hemisfério norte. Passo 3: então qual é a causa real? A inclinação do eixo de rotação da Terra em relação ao plano de sua órbita, que é de 23,5 graus. Passo 4: como essa inclinação causa estações? Durante parte do ano, o hemisfério norte está inclinado em direção ao Sol. Passo 5: consequências disso. Os raios solares incidem mais diretamente sobre o hemisfério norte, com menor ângulo de incidência. Passo 6: mais consequências. Os dias ficam mais longos no hemisfério inclinado para o Sol. Passo 7: resultado. O hemisfério norte recebe mais energia solar por unidade de superfície e por mais horas ao dia, resultando em verão. Passo 8: o hemisfério oposto. Enquanto o norte tem verão, o sul está inclinado para longe do Sol e tem inverno. Passo 9: os equinócios. Duas vezes por ano, o eixo da Terra fica perpendicular à linha que vai da Terra ao Sol, e ambos os hemisférios recebem igual quantidade de luz. Passo 10: regiões tropicais. Próximo ao equador, o Sol está sempre alto no céu e as variações sazonais são menores, com estações seca e chuvosa em vez de quente e fria. Passo 11: confirmar que a distância ao Sol tem efeito mínimo comparado à inclinação. A excentricidade da órbita terrestre afeta apenas cerca de 7% da intensidade solar ao longo do ano. Passo 12: formular resposta clara. [/Pensando]
Resposta: As estações do ano são causadas pela inclinação do eixo de rotação da Terra em relação ao plano de sua órbita, que é de cerca de 23,5 graus, e não pela variação de distância ao Sol. Quando o hemisfério norte está inclinado em direção ao Sol, os raios solares incidem mais diretamente e os dias são mais longos, resultando em verão. Ao mesmo tempo, o hemisfério sul está inclinado para longe do Sol e tem inverno. Seis meses depois a situação se inverte. Nos equinócios, os dois hemisférios recebem igual quantidade de luz solar, resultando em dias e noites de igual duração.

Pergunta: O que causa os terremotos?
[Pensando] Passo 1: contextualizar com a estrutura da Terra. A Terra tem crosta, manto, núcleo externo líquido e núcleo interno sólido. Passo 2: a teoria central. Tectônica de placas: a litosfera está dividida em placas que se movem sobre o manto semi-fluido. Passo 3: o que causa o movimento? Correntes de convecção no manto, movidas pelo calor do núcleo, arrastam as placas. Passo 4: o que acontece nas bordas das placas? As placas interagem de três formas principais. Passo 5: primeira forma: subducção. Uma placa densa mergulha sob outra. A placa que subducta pode raspar e causar sismos. Passo 6: segunda forma: colisão. Duas placas continentais colidem e dobram, formando montanhas como os Himalaias, com frequentes terremotos. Passo 7: terceira forma: falha transformante. Duas placas deslizam horizontalmente uma em relação à outra, como na falha de San Andreas na Califórnia. Passo 8: por que os terremotos ocorrem em zonas específicas? Porque as bordas das placas são onde as tensões acumuladas são liberadas. Passo 9: o mecanismo direto. As rochas nas bordas das placas se deformam elasticamente acumulando energia por décadas ou séculos. Quando a tensão supera o limite de ruptura da rocha, ocorre o deslizamento súbito que libera energia em forma de ondas sísmicas. Passo 10: o que são ondas sísmicas? Vibrações que se propagam pela Terra a partir do foco do terremoto. Passo 11: o epicentro é o ponto na superfície diretamente acima do foco. Passo 12: o que causa o tsunami? Terremotos submarinos que deslocam verticalmente grandes massas de água. Passo 13: sintetizar resposta. [/Pensando]
Resposta: Terremotos são causados pelo movimento das placas tectônicas, os grandes fragmentos em que a litosfera terrestre está dividida e que se movem lentamente impulsionados por correntes de convecção no manto. Nas bordas das placas, as rochas acumulam tensão elástica ao longo de décadas ou séculos. Quando essa tensão supera o limite de ruptura, ocorre o deslizamento súbito da falha geológica, liberando energia em forma de ondas sísmicas que sacudem a superfície. Colisões, subducções e falhas transformantes como a de San Andreas são as principais zonas sísmicas. Terremotos submarinos que deslocam verticalmente a coluna d'água podem gerar tsunamis.

Pergunta: Como o cérebro processa emoções?
[Pensando] Passo 1: identificar o sistema relevante. O sistema límbico é a região cerebral principal para processamento emocional. Passo 2: estruturas principais. Amígdala, hipocampo, córtex cingulado anterior, hipotálamo e núcleo accumbens. Passo 3: o papel da amígdala. É o centro de detecção e resposta a estímulos emocionalmente relevantes, especialmente ameaças e recompensas. Detecta estímulos em milissegundos antes que a consciência os processe. Passo 4: resposta de luta ou fuga. Quando a amígdala detecta ameaça, ativa o hipotálamo que desencadeia resposta do sistema nervoso simpático: adrenalina, aceleração cardíaca, dilatação de pupila, tensão muscular. Passo 5: o papel do córtex pré-frontal. Região de controle executivo que modula as respostas da amígdala, permitindo avaliação racional da situação em vez de reação impulsiva. É a base da regulação emocional. Passo 6: emoções básicas. Paul Ekman identificou seis emoções básicas com expressões faciais universais: alegria, tristeza, medo, raiva, nojo e surpresa. Passo 7: neurotransmissores. Dopamina está ligada ao sistema de recompensa e motivação. Serotonina regula humor, sono e apetite. Norepinefrina é ativada no estado de alerta. Oxitocina promove vínculos sociais e confiança. Passo 8: como a experiência emocional chega à consciência? Sinal da amígdala vai para o córtex, que gera a experiência consciente da emoção combinando o estímulo com memórias, contexto e estado corporal. Passo 9: emoções e memória. A amígdala modula a consolidação de memórias no hipocampo. Eventos emocionais são lembrados com mais vividez e detalhe. Passo 10: regulação emocional. A cognição pode regular emoções: reavaliação cognitiva, onde reinterpretamos o significado de um evento, é a estratégia mais eficaz de regulação emocional. Passo 11: síntese. [/Pensando]
Resposta: As emoções são processadas principalmente pelo sistema límbico do cérebro. A amígdala detecta estímulos emocionalmente relevantes, especialmente ameaças, em milissegundos e pode desencadear respostas físicas automáticas como aceleração cardíaca antes que a consciência processe o evento. O hipotálamo ativa o sistema nervoso simpático para a resposta de luta ou fuga. O córtex pré-frontal modula e regula as respostas da amígdala, permitindo avaliação racional em vez de reação impulsiva. Neurotransmissores como dopamina, serotonina e oxitocina modulam diferentes aspectos do estado emocional. Eventos emocionalmente intensos são lembrados mais vividamente porque a amígdala reforça sua consolidação no hipocampo.
"""

# Corpus completo e corpus por fase (local)
CORPUS = CORPUS_TEXTO + "\n\n" + CORPUS_QA + "\n\n" + CORPUS_THINK
CORPUS_POR_FASE = [CORPUS_TEXTO, CORPUS_QA, CORPUS_THINK]

# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS EXTERNOS — conversas reais do HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_conversa(turnos: list) -> str:
    """Converte lista de turnos {'role', 'content'} em texto de treino."""
    partes = []
    for t in turnos:
        role = t.get("role", "")
        txt  = (t.get("content") or t.get("text") or "").strip()
        if not txt:
            continue
        if role in ("user", "prompter", "human"):
            partes.append(f"Pergunta: {txt}")
        elif role in ("assistant", "gpt", "bot"):
            partes.append(f"Resposta: {txt}")
    return "\n".join(partes)


def carregar_datasets_externos(max_exemplos_por_ds: int = 3000,
                                verbose: bool = True) -> str:
    """
    Carrega e converte datasets reais de conversação em português.
    Retorna texto concatenado pronto para fine-tuning.
    Cada dataset tem tratamento de erro independente — se um falhar, os outros
    continuam normalmente.

    Parâmetros:
        max_exemplos_por_ds: máximo de exemplos usados por dataset
        verbose: se True, imprime progresso

    Retorna:
        str com todas as conversas formatadas
    """
    try:
        from datasets import load_dataset
    except ImportError:
        if verbose:
            print("  [aviso] biblioteca 'datasets' nao instalada. Pulando datasets externos.")
        return ""

    textos = []

    # ── 1. OpenAssistant oasst1 — conversas anotadas por humanos ─────────────
    try:
        if verbose:
            print("  Carregando OpenAssistant/oasst1 (PT)...")
        # oasst1 tem splits "train" e "validation"
        ds_train = load_dataset("OpenAssistant/oasst1", split="train")
        ds_val   = load_dataset("OpenAssistant/oasst1", split="validation")

        # Combina ambos os splits e filtra por português
        todas = list(ds_train) + list(ds_val)
        pt    = [r for r in todas if r.get("lang", "") == "pt"]

        # Reconstrói pares prompter → assistant a partir do parent_id
        msgs_idx = {row["message_id"]: row for row in pt}
        pares = []
        for row in pt:
            if row.get("role") == "assistant" and row.get("parent_id") in msgs_idx:
                pai = msgs_idx[row["parent_id"]]
                if pai.get("role") == "prompter":
                    perg = pai["text"].strip()
                    resp = row["text"].strip()
                    if perg and resp:
                        pares.append({"pergunta": perg, "resposta": resp})

        for p in pares[:max_exemplos_por_ds]:
            textos.append(f"Pergunta: {p['pergunta']}\nResposta: {p['resposta']}")

        if verbose:
            print(f"    {len(pares)} pares PT carregados do oasst1")
    except Exception as e:
        if verbose:
            print(f"  [aviso] oasst1 falhou: {e}")

    # ── 2. Everyday Conversations — chat cotidiano básico ─────────────────────
    try:
        if verbose:
            print("  Carregando everyday-conversations...")
        # Split correto e "train_sft" (nao "train")
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k",
                          split="train_sft")
        for row in list(ds)[:max_exemplos_por_ds]:
            msgs = row.get("messages") or row.get("conversation") or []
            if msgs:
                txt = _fmt_conversa(msgs)
                if txt.strip():
                    textos.append(txt)
        if verbose:
            print(f"    {min(len(ds), max_exemplos_por_ds)} conversas carregadas")
    except Exception as e:
        if verbose:
            print(f"  [aviso] everyday-conversations falhou: {e}")

    # ── 3. lmsys-chat-1m — requer login HF (gated dataset) ──────────────────
    # Para habilitar: faça login com `huggingface-cli login` e aceite os termos
    # em https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        if verbose:
            print("  [info] lmsys-chat-1m ignorado (requer HF_TOKEN — dataset gated)")
    else:
        try:
            if verbose:
                print("  Carregando lmsys/lmsys-chat-1m (PT) via streaming...")
            ds = load_dataset("lmsys/lmsys-chat-1m", split="train",
                              streaming=True, token=hf_token)
            contagem = 0
            for row in ds:
                if contagem >= max_exemplos_por_ds:
                    break
                if row.get("language", "") != "pt":
                    continue
                txt = _fmt_conversa(row.get("conversation", []))
                if txt.strip():
                    textos.append(txt)
                    contagem += 1
            if verbose:
                print(f"    {contagem} conversas PT carregadas do lmsys-chat-1m")
        except Exception as e:
            if verbose:
                print(f"  [aviso] lmsys-chat-1m falhou: {e}")

    resultado = "\n\n".join(textos)
    if verbose:
        print(f"  Total externo: {len(textos)} conversas  |  {len(resultado):,} chars\n")
    return resultado
