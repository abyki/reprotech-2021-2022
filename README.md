# reprotech-2021-2022

Ссылка на отчет https://docs.google.com/document/d/1CDnAZS8Cjva9GVXuYOal5E9Da3AGjQD7WGBtKpNupMI/edit?usp=sharing

Ссылка на колаб https://colab.research.google.com/github/abyki/reprotech-2021-2022/blob/main/NLG-%D0%A25.ipynb

Генерация текста из структурированных данных

Акаев М.Х. 


В быстро меняющемся мире человеку важно также быстро осваивать новую информацию. Но информация, хранящаяся в структурированном формате, не всегда понятна человеку. Чтобы донести информацию до конечного пользователя специалисты описывают полученные данные, будь то результаты анализа или же климатические данные, простым понятным человеку текстом. Но на это уходит очень много времени. Автоматизация данного процесса улучшит эффективность работы.
Недостаточная воспроизводимость является одной из актуальных проблем в науках о данных. Data-to-text generation преобразует информацию из структурированного формата, например таблица, в естественный язык. Это позволяет читать или прослушивать структурированную информацию, например, когда устройство отображает прогноз погоды или голосовой помощник отвечает на вопрос.
Решается задача генерации текстовых описаний и отчетов по структурированным данным в применении к данным экспериментов в машинном обучении. Это позволит улучшить Научную коммуникацию для тех кто выполняет эксперимент.

·       Собранные данные для построения отчета.
·       Обученная модель генерации текста

·       Текст на естественном языке, поясняющий постановку и процедуру выполнения эксперимента

Структурированные данные описывают какое-то событие, которую модель, в свою очередь, должна описать на понятном человеку естественном языке. Генерация текста по данным уже исследовалась ранее и есть несколько моделей посвященные этой задаче. Например, A Hierarchical Model for Data-to-Text Generation[1] и Plan-then-Generate: Controlled Data-to-Text Generation via Planning[2]. Также, в системе mldev[3] эксперимент уже описывается в виде объектной структуры (графа), который пользователь заполняет параметрами, и собираются данные о ходе эксперимента для построения графиков и дальнейшего анализа.

Алгоритм предлагается использовать следующий:
Эксперимент проводится в системе mldev
Из системы извлекается граф описания эксперимента
Граф подается на вход обученной модели для генерации текстового описания
Это описание в купе с построенными графиками формируют небольшой отчет
      
Предполагается использование датасета статей по машинному обучению, доработка и дообучение уже существующей модели для задачи создания описания эксперимента

Данные:
WebNLG - датасет троек RDF из 16 различных категорий DBpedia: Airport, Astronaut, Building, City, ComicsCharacter, Food, Monument, SportsTeam, University, and WrittenWork, Athlete, Artist, CelestialBody, MeanOfTransportation, Politician.
ToTTo - датасет с более чем 120000 парами данных и соответствующего текста из википедии
RotoWire - датасет из таблиц статистики более чем 4000 баскетбольных игр с паре с журналистским описанием этих игр
