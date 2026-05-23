"""
Синтетичний корпус для калібрування детектора.

Корпус використовується скриптом `s5_train_calibrator.py` для навчання
логістичного калібратора. Він охоплює дві групи прикладів:

  • людські академічні тексти (укр + англ) — формальний регістр,
    варіативні абзаци, нерівномірне використання конекторів, авторські
    неоднорідності, типові для природного письма;

  • тексти сучасних великих мовних моделей (Claude Opus, GPT-4+,
    Gemini) — характерний стиль із hedging, паралельними конструкціями,
    тріплет-перерахуваннями «X, Y, Z», симетричними абзацами та
    рівномірною щільністю абстрактного словника.

Тексти подаються в оригінальних мовах. Розширені стилометричні ознаки,
що використовує калібратор, обчислюються двомовно, тож переклад
не потрібен.
"""

# --- Людські академічні тексти (українською) -------------------------------
# Стиль: курсові, дипломні, наукові статті українських авторів.
# Характерні ознаки: формальний регістр, варіативні абзаци,
# нерівномірне використання конекторів, авторські неоднорідності.

HUMAN_UK_ACADEMIC = [
    """Інформаційна система реалізується на основі клієнт-серверної архітектури, яка передбачає розподіл функціональних компонентів між клієнтською частиною (інтерфейсом користувача) та серверною частиною (логікою обробки запитів і доступу до бази даних). Такий підхід дозволяє централізувати керування даними, зменшити навантаження на клієнта, а також забезпечити масштабованість і модульність системи. У межах клієнт-серверної моделі реалізовано Zero-Trust архітектуру, що ґрунтується на принципі «нікому не довіряй, завжди перевіряй». Усі запити, що надходять від клієнта, проходять автентифікацію та авторизацію, незалежно від їхнього джерела. Сервер не вважає довіреним жодного клієнта без валідації запиту за допомогою токенів доступу.""",

    """Метод скінченних елементів є одним з найбільш поширених чисельних методів розв'язування крайових задач математичної фізики. Його суть полягає в розбитті області визначення розв'язку на скінченну кількість простих підобластей, на кожній з яких шуканий розв'язок апроксимується відомими базисними функціями. У результаті початкову диференціальну задачу зводять до системи лінійних алгебраїчних рівнянь відносно вузлових значень розв'язку. На практиці точність методу залежить від вибору сітки розбиття та степеня поліномів, що використовуються як базис.""",

    """Дослідження виконано на матеріалі сорока корпоративних звітів за 2018–2022 роки. Вибірка формувалась з компаній фінансового сектору, які працюють на ринку України не менше п'яти років. Збір даних здійснювався вручну з публічних джерел: офіційних сайтів компаній та системи розкриття інформації НКЦПФР. Отримані тексти оброблялись інструментами автоматичного морфологічного аналізу, що дозволило виокремити лексичні маркери стратегічного дискурсу. Ми навмисно обмежили вибірку лише українськими емітентами, оскільки мовно-культурний контекст істотно впливає на вибір аргументаційних стратегій.""",

    """Електрохімічна імпедансна спектроскопія (EIS) — потужний метод дослідження властивостей електродних матеріалів. У роботі застосовано модель Рендлеса з елементом постійної фази для опису поведінки електрод-електролітного інтерфейсу. Експерименти проводились у частотному діапазоні від 0.01 Гц до 100 кГц при температурі 25 °C. Аналіз годографів Найквіста дозволив розділити внески обмеженого дифузією та активаційного контролю. Окремо слід відзначити, що при низьких частотах спостерігалось відхилення від ідеальної поведінки, яке ми пов'язуємо з пористою структурою електрода.""",

    """Питання правового регулювання штучного інтелекту в Європейському Союзі набуло особливої актуальності після ухвалення AI Act у 2024 році. Регламент запроваджує ризик-орієнтований підхід: системи високого ризику зобов'язані проходити процедуру оцінки відповідності, тоді як для систем мінімального ризику передбачено лише добровільні кодекси поведінки. Натомість на національному рівні країни-члени ЄС досі демонструють фрагментарне впровадження, що породжує неузгодженості. Цей розрив між загальноєвропейською рамкою та національними практиками створює простір для регуляторного арбітражу.""",

    """Археологічні розкопки в околицях Києва, проведені експедицією у 2019 році, дали несподівані результати. На глибині близько метра було виявлено фрагменти кераміки, які за технологічними ознаками належать до черняхівської культури. Це не вписувалось у попередню реконструкцію розселення населення в цій місцевості. Особливо цікавим виявився виявлений у північній частині розкопу комплекс залізних виробів — серпи, ножі, кільця — що мають аналогії на пам'ятках Подніпров'я.""",

    """Хвороба Альцгеймера лишається однією з нерозв'язаних медичних проблем. Основна гіпотеза патогенезу — амілоїдна каскадна — пояснює нейродегенерацію накопиченням бета-амілоїдних бляшок у корі головного мозку. Проте останні клінічні випробування препаратів-інгібіторів секретази не дали очікуваного клінічного ефекту, що поставило цю гіпотезу під сумнів. Альтернативні концепції, зокрема тау-гіпотеза та запальна теорія, наразі також не отримали остаточного підтвердження.""",

    """Аналіз показників економічного зростання країн Південно-Східної Азії у період 2010–2020 років виявив істотну гетерогенність. Сінгапур і Південна Корея зберігали стабільне зростання на рівні 3–4% ВВП щорічно, тоді як Філіппіни та В'єтнам демонстрували вищі темпи — до 6–7%, проте зі значно більшою волатильністю. Причини такої розбіжності полягають у структурних особливостях економік. Дві перші країни мають розвинений високотехнологічний сектор, експорт якого менш чутливий до коливань сировинних цін.""",

    """Питання про природу свідомості залишається відкритим у філософії розуму. Картезіанський дуалізм, який панував у європейській традиції до XX століття, поступився місцем різноманітним матеріалістичним концепціям — від теорії тотожності до функціоналізму. Втім, жодна з них не дає переконливого розв'язання так званої «важкої проблеми» Чалмерса. Як з'ясовується, навіть повний нейронаукових опис мозкових процесів не пояснює, чому ці процеси супроводжуються суб'єктивним переживанням.""",

    """Методика навчання іноземних мов зазнала суттєвих змін за останні два десятиліття. На зміну граматико-перекладному методу прийшов комунікативний підхід, який зосереджується на здатності учня використовувати мову в реальних ситуаціях спілкування. Сучасні підручники будуються навколо тематичних розділів, у яких граматичні структури вводяться функціонально. Проте практика українських шкіл показує, що повний відхід від традиційних граматичних вправ не завжди виправданий — особливо при навчанні мов з відмінним типом граматики.""",

    """Виявлення фальсифікації цифрових зображень — задача, важливість якої зросла з розвитком технологій генеративного машинного навчання. Класичні підходи, основані на аналізі шуму сенсора (PRNU), залишаються ефективними проти традиційних маніпуляцій типу copy-paste. Проте проти deepfake-зображень вони безсилі, оскільки генеративні моделі створюють зображення з нуля. У роботі пропонується гібридний підхід: поєднання частотного аналізу артефактів GAN-генерації зі стандартним PRNU-методом. Це дозволяє детектувати як модифікації автентичних знімків, так і повністю синтезовані зображення.""",

    """Музеї сучасного мистецтва в Україні переживають важливий етап інституційного становлення. Якщо в 1990-х роках їхня діяльність була переважно ентузіастською і трималась на окремих особистостях, то протягом 2010-х сформувалась мережа професійних установ зі сталим фінансуванням. PinchukArtCentre, Мистецький Арсенал, IZONE — кожна з цих інституцій має власну стратегію роботи з аудиторією. Проте війна, яка триває з 2022 року, поставила перед сектором нові виклики: евакуація колекцій, цифровізація фондів, переосмислення кураторської практики в умовах постійної загрози.""",

    """Алгоритм Сімплекс-методу, запропонований Данцигом 1947 року, протягом десятиліть лишався основним інструментом розв'язання задач лінійного програмування. Його обчислювальна складність у найгіршому випадку експоненційна, однак на практиці метод демонструє поліноміальну поведінку. Перші поліноміально-часові алгоритми для ЛП — метод еліпсоїдів Хачіяна 1979 року та метод внутрішньої точки Кармаркара 1984 року — стали справжнім проривом. Незважаючи на теоретичні переваги, в індустріальних застосуваннях Сімплекс-метод досі залишається конкурентоспроможним завдяки ефективним імплементаціям.""",

    """Класифікація лишайників — одна з найскладніших задач сучасної ботаніки. На відміну від рослин, лишайники є симбіотичними утвореннями: гриб-мікобіонт і водорость-фотобіонт існують як єдиний організм. Це ставить під сумнів класичну біологічну концепцію виду. Молекулярно-філогенетичні дослідження останнього десятиліття показали, що в межах одного «виду» лишайника часто співіснують кілька генетично різних штамів гриба. Деякі дослідники взагалі пропонують відмовитись від традиційної таксономії на користь полігенетичної.""",

    """Феномен «вигорання» (burnout) серед медичних працівників набув особливо тривожних масштабів під час пандемії COVID-19. Дослідження, проведене серед лікарів інтенсивної терапії у дев'яти європейських країнах протягом 2020–2021 років, показало, що понад 60% респондентів демонстрували клінічно значущі симптоми емоційного виснаження. Найвищі показники спостерігались у країнах з обмеженими ресурсами охорони здоров'я та слабкою системою психологічної підтримки персоналу. Слід окремо наголосити, що дані до пандемії в тих самих країнах також не були добрими — пандемія лише загострила хронічну проблему.""",

    """Технологія блокчейн часто подається в популярних джерелах як засіб «децентралізації всього». На практиці більшість реальних застосувань блокчейну страждають від суттєвої внутрішньої суперечності. Хоча сам протокол справді є децентралізованим, інфраструктурні компоненти — біржі, обчислювальні потужності для майнінгу, основні розробницькі команди — концентруються в руках обмеженого кола гравців. Це створює парадокс: децентралізована технологія обслуговує цілком централізовану економічну структуру.""",
]

# --- Людські академічні тексти (англійською) -------------------------------

HUMAN_EN_ACADEMIC = [
    """The persistence of social inequality in post-industrial economies has proven resistant to standard policy interventions. Welfare expansion in Scandinavian countries during the 1970s reduced gini coefficients substantially but did not eliminate intergenerational transmission of disadvantage. More recent comparative studies, particularly the work of Chetty and colleagues on the United States, suggest that residential segregation plays a role independent of household income. What emerges from this literature is a picture in which spatial concentration of poverty creates self-reinforcing dynamics that pure redistribution cannot break.""",

    """Crystallographic analysis of the recovered sample revealed an unexpected phase composition. While we had anticipated predominant monoclinic structure based on the synthesis temperature, X-ray diffraction patterns indicated approximately 30% tetragonal phase content. We initially attributed this to insufficient annealing time, but extending the annealing protocol by 4 hours did not eliminate the secondary phase. After consultation with the original authors of the synthesis procedure, we determined that trace iron contamination from the alumina crucible was stabilizing the tetragonal structure — something that had not been documented in the original publication.""",

    """The historical record of monastic communities in early medieval Ireland is preserved unevenly. While Iona and Lindisfarne have left extensive written documentation, smaller foundations such as those at Clonmacnoise and Glendalough are known primarily through archaeological evidence and later hagiographical accounts of varying reliability. This asymmetry has shaped historiographical traditions in ways that may distort our understanding of the period. Recent reassessments by Charles-Edwards and others have argued that the dominance of certain narratives reflects later medieval political concerns rather than the actual relative importance of these centres.""",

    """Mosquito-borne diseases pose increasing challenges to public health in temperate zones as a consequence of changing climate patterns. Aedes albopictus, originally a southeast Asian species, has now established stable populations in southern Europe and parts of North America. Surveillance data collected by ECDC between 2015 and 2023 documents northward expansion at approximately 150 kilometres per decade. The implications for transmission of dengue and chikungunya viruses in regions previously considered low-risk are substantial, though the epidemiological response has been uneven across jurisdictions.""",

    """Quantum error correction remains the fundamental obstacle to practical quantum computation. Current state-of-the-art superconducting qubits achieve coherence times in the hundreds of microseconds, which is insufficient for fault-tolerant computation at meaningful scales. The surface code, proposed by Kitaev and refined extensively over the past two decades, offers a theoretical pathway to fault tolerance but at the cost of substantial physical-qubit overhead. Recent demonstrations by Google and IBM of logical qubits with error rates below the physical threshold are encouraging, though scaling these results to thousands of logical qubits will require advances in fabrication that have not yet materialized.""",

    """Linguistic borrowing in Modern Ukrainian has followed a complex trajectory shaped by political and cultural pressures. The lexical influence of Russian during the Soviet period was extensive, particularly in technical and administrative vocabulary. Since 1991, and especially after 2014, there has been a deliberate effort — both organized through institutional standardization bodies and emergent in everyday speech — to replace Russian-origin terms with Ukrainian alternatives, often through reactivation of dialectal or historical vocabulary. The success of these efforts varies considerably across registers and lexical fields.""",

    """The recovery of damaged paintings using non-invasive techniques has progressed substantially since the introduction of macro X-ray fluorescence imaging in the 2010s. This method allows mapping of elemental distributions across an entire canvas without sample removal, revealing underdrawings and pentimenti that traditional examination methods could not detect. Application to several major works in European collections has produced unexpected findings — particularly in cases where canvases had been heavily restored in the 19th century and the modifications were not documented. However, interpretation of these maps requires considerable expertise and the technology is not yet widely accessible to smaller institutions.""",

    """Field observations of corvid behaviour in urban environments have raised interesting questions about animal cognition. Crows in Tokyo have been documented using traffic patterns to crack walnuts — placing them on roads where they will be run over, then waiting for traffic lights to change before retrieving the kernel. Whether this constitutes genuine tool use in the cognitive sense, or sophisticated trial-and-error learning, remains contested. The methodological difficulties of distinguishing these possibilities in wild populations have so far prevented definitive conclusions.""",
]

# --- Modern-LLM тексти (стиль Claude Opus 4.x / GPT-4+ / Gemini) -----------
# Стиль: hedging, паралелізми, тріплет-перерахування, симетричні абзаци,
# рівні довжини речень, високий формальний регістр БЕЗ зловживання
# "However/Furthermore". Це той стиль, який не ловить базова модель.

AI_MODERN_UK = [
    """Архітектура мікросервісів забезпечує гнучкість, масштабованість і простоту супроводу великих програмних систем. На відміну від монолітного підходу, де всі функціональні компоненти тісно пов'язані, мікросервісна модель передбачає розділення системи на незалежні служби — кожна з власною відповідальністю, власною базою даних і власним життєвим циклом розгортання. Такий підхід дозволяє командам розробки працювати паралельно, оновлювати окремі компоненти без ризику для системи в цілому та масштабувати лише ті частини, де це справді потрібно. Водночас мікросервісна архітектура вимагає зрілої інфраструктури: системи оркестрації, моніторингу, розподіленого трейсингу та механізмів узгодження даних між службами.""",

    """Підбір алгоритму машинного навчання для конкретної задачі залежить від кількох ключових факторів — обсягу доступних даних, природи цільової змінної та вимог до інтерпретованості результату. Для задач класифікації на невеликих, добре структурованих наборах даних логістична регресія залишається сильною базовою лінією завдяки своїй простоті, обчислювальній ефективності та можливості інтерпретації коефіцієнтів. Якщо ж дані мають складну нелінійну структуру, доцільно розглянути методи на основі дерев — випадковий ліс або градієнтний бустинг — які зазвичай демонструють кращу точність при помірних обчислювальних витратах. Для задач з великими обсягами даних та складною семантичною структурою найвищу точність зазвичай досягають глибокі нейронні мережі, проте за рахунок суттєвого зростання обчислювальних вимог.""",

    """Сучасні системи кешування побудовані на трьох фундаментальних принципах — локальності звертань, ієрархічності рівнів та узгодженості даних між кешем та основним сховищем. Принцип локальності дозволяє ефективно прогнозувати, які дані будуть потрібні найближчим часом, ґрунтуючись на статистиці попередніх звертань. Ієрархічна організація — від швидких кешів першого рівня до повільніших, але об'ємніших — забезпечує оптимальний баланс між швидкістю доступу та обсягом збережених даних. Принципи узгодженості, своєю чергою, регламентують, як саме оновлюються кешовані копії при зміні первинних даних, що особливо важливо в розподілених системах. Поєднання цих трьох принципів визначає продуктивність системи в цілому.""",

    """Цифрова трансформація освіти охоплює три взаємопов'язані напрями — інфраструктурне забезпечення, методологічне оновлення та формування цифрової грамотності всіх учасників навчального процесу. Інфраструктурний рівень передбачає розгортання сучасних навчальних платформ, інтеграцію систем управління знаннями та забезпечення стабільного доступу до інтернету. Методологічне оновлення стосується способів подачі матеріалу: гібридні формати, інтерактивні елементи, адаптивні системи навчання — кожен з цих підходів вимагає переосмислення ролі викладача. Цифрова грамотність, своєю чергою, формується не одномоментно, а через систематичне залучення учнів і педагогів до роботи з відповідними інструментами.""",

    """Принцип роботи блокчейну ґрунтується на трьох ключових концепціях — криптографічному хешуванні, розподіленому консенсусі та незмінності записів. Хешування забезпечує цілісність кожного блоку: будь-яка зміна вмісту блоку призводить до зміни його криптографічного відбитка, що одразу стає помітним. Розподілений консенсус — найчастіше у формі Proof-of-Work або Proof-of-Stake — гарантує, що жоден окремий учасник не може одноосібно внести зміни до спільного реєстру. Незмінність записів є логічним наслідком перших двох принципів: оскільки кожен блок посилається на хеш попереднього, спроба змінити старі записи вимагала б повторного обчислення всіх наступних блоків. Сукупність цих властивостей робить блокчейн придатним для застосувань, де потрібна довіра без посередників.""",

    """Підходи до забезпечення кібербезпеки сучасних організацій можна умовно розділити на три категорії — превентивні, детективні та реагувальні. Превентивні заходи спрямовані на запобігання інцидентам: налаштування міжмережевих екранів, контроль доступу, регулярне оновлення програмного забезпечення. Детективні механізми, навпаки, забезпечують виявлення вже наявних загроз — системи виявлення вторгнень, аналіз журналів подій, інструменти поведінкової аналітики. Реагувальні процеси активуються після виявлення інциденту: ізоляція компрометованих ресурсів, відновлення даних, форензичний аналіз. Зрілість безпекової практики організації зазвичай оцінюється за тим, наскільки збалансовано розвинуті всі три категорії.""",

    """Контейнеризація змінила підхід до розгортання програмного забезпечення завдяки трьом ключовим перевагам — переносимості, ізоляції та ефективності використання ресурсів. Переносимість досягається за рахунок упаковки застосунку разом з усіма його залежностями в єдиний образ, який однаково запускається в будь-якому середовищі. Ізоляція гарантує, що процеси одного контейнера не впливають на інші, що особливо важливо в багатотенантних сценаріях. Ефективність полягає в значно меншому накладному навантаженні порівняно з повноцінною віртуалізацією — контейнери ділять ядро операційної системи, що дозволяє запускати на одному вузлі десятки або навіть сотні ізольованих застосунків. Поєднання цих властивостей зробило контейнери стандартом галузі за останнє десятиліття.""",

    """Аналіз ефективності маркетингових кампаній зазвичай ґрунтується на трьох групах показників — охоплення, залучення та конверсії. Охоплення відображає кількість унікальних осіб, які побачили рекламне повідомлення, та є базовим показником видимості бренду. Залучення характеризує якісну взаємодію — час, проведений на сторінці, кількість переглянутих елементів, частоту повернень. Конверсія, своєю чергою, фіксує цільові дії: реєстрації, замовлення, підписки. Кожна з груп показників має власну роль, і зосередження лише на одній з них зазвичай призводить до спотвореної картини. Збалансоване відстеження всіх трьох рівнів дозволяє маркетинговій команді ухвалювати рішення на основі повної воронки.""",

    """Технології обробки природної мови розвиваються переважно у трьох напрямах — розпізнавання структури, моделювання семантики та генерація відповідей. Розпізнавання структури охоплює задачі морфологічного, синтаксичного та частково семантичного аналізу — це базовий рівень, що дозволяє системі розуміти будову вхідного тексту. Моделювання семантики виходить за межі окремих речень: тут вирішуються задачі визначення тематики, виявлення емоційного забарвлення, побудови знаннєвих графів. Генерація відповідей — найбільш динамічний напрям останніх років — забезпечує створення зв'язних, контекстуально доречних текстів у відповідь на запити користувача. Сучасні системи зазвичай поєднують всі три напрями в єдиній архітектурі.""",

    """Дослідження поведінки споживачів у цифровому середовищі ґрунтується на трьох основних джерелах даних — поведінкових журналах, опитувальних даних та біометричних показниках. Поведінкові журнали фіксують реальні дії користувачів — переходи, кліки, час перебування на сторінках — і дозволяють відтворити шлях клієнта без покладання на самозвіт. Опитувальні дані додають контекст: мотивацію, очікування, оцінку досвіду. Біометричні показники — рух очей, шкіряно-гальванічна реакція, частота серцевих скорочень — розкривають емоційний вимір взаємодії. Поєднання цих джерел створює багатовимірну картину, що значно перевершує можливості будь-якого окремого методу.""",
]

# Додаткові людські тексти — спеціально схожі за тематикою на LLM-тексти,
# щоб калібратор навчився відрізняти НЕ за тематикою, а за стилем. Це
# критично: без них модель може випадково запам'ятати "тема = техніка → ШІ".
HUMAN_UK_TECHNICAL = [
    """Розробка нашого внутрішнього інструменту моніторингу зайняла більше часу, ніж планувалось. Спочатку ми хотіли просто обгорнути Prometheus у зручніший інтерфейс, але виявилось, що команді не вистачає компетенції з PromQL. Через два місяці зрозуміли, що краще було взяти Datadog — економія часу інженерів перевершила б ліцензійні витрати. Тепер маємо half-baked рішення, яким користуються троє людей з шести. Висновок очевидний: не всі open-source альтернативи виправдані.""",

    """Питання вибору архітектури — мікросервіси чи моноліт — у нашому проєкті вирішувалось довго. Ми зрештою обрали моноліт, хоча модне рішення підказувало інше. Причина проста: команда з п'ятьох розробників не потягне операційні витрати мікросервісної інфраструктури. Уся ця історія з контейнерами, Kubernetes, service mesh — це непосильний оверхед для невеликого продукту. Через рік не пошкодували: швидкість розробки висока, проблем з деплоєм майже немає.""",

    """Кешування в нашому застосунку реалізовано доволі примітивно — на рівні Redis з простими key-value. Спочатку я хотів зробити щось більш складне з ієрархіями кешів, інвалідацією за тегами тощо, але після обговорення з командою вирішили, що оверінжиніринг тут зайвий. Зараз ця реалізація обслуговує близько ста тисяч запитів на день і жодного разу не була вузьким місцем. Інколи проста реалізація — найкраща, навіть якщо здається занадто грубою для досвідченого інженера.""",

    """Працюючи з квантовими алгоритмами в академічному середовищі, я постійно стикаюсь з невідповідністю між заявами в популярних статтях та реальним станом галузі. У публічних дискусіях часто говорять про «квантову перевагу», тоді як на практиці ми ледь утримуємо когерентність на десятці кубітів. NISQ-епоха не дала майже жодних практично корисних результатів попри багатомільярдні інвестиції. Звичайно, фундаментальні дослідження мають свою цінність, але це не те, що очікують від нас фінансувальники.""",

    """В моїй практиці база даних PostgreSQL виявилась найнадійнішим вибором для більшості проєктів. Колись я захоплювався MongoDB, але після кількох інцидентів з втратою даних повернувся до традиційного реляційного підходу. PostgreSQL з JSONB полями покриває майже всі потреби, які раніше здавались специфічними для NoSQL. Звичайно, є ніші — масштабована аналітика, документне зберігання — де спеціалізовані БД виправдані. Але стартовий вибір майже завжди має бути на користь PostgreSQL.""",

    """Інтерфейс нашого мобільного додатку перероблявся тричі за два роки. Перший дизайн був занадто складним — користувачі скаржились на купу прихованих жестів. Другий, спрощений, виявився надто стерильним і не зачіпав емоційно. Третій варіант, з яким працюємо зараз, поєднує мінімалізм у структурі з персоналізованими деталями. Метрики залученості зросли на 40%, але я досі не впевнений, що ми знайшли остаточне рішення. UX — це постійна еволюція, а не одноразова робота.""",

    """Безпека наших корпоративних систем будувалася поетапно й часто реактивно. Перший серйозний поштовх стався після фішингової атаки 2019 року, коли в нас вкрали доступ до облікового запису фінансового директора. Тоді ми впровадили обов'язкову двофакторну автентифікацію — і це було наше найкраще рішення. Решта механізмів — SIEM, EDR, навчання персоналу — додавалися поступово. Чесно кажучи, до повноцінного Zero-Trust ми ще далеко, але рух у правильному напрямку є.""",

    """Машинне навчання в реальному виробництві відрізняється від академічних статей у двох ключових моментах: дані ніколи не такі чисті, як здається, а продуктивність моделі — це лише початок. На моєму поточному проєкті ми витратили шість місяців на побудову інфраструктури MLOps — пайплайни даних, моніторинг дрейфу, A/B-тестування — і лише два тижні на саму модель. Парадокс полягає в тому, що звичайна логістична регресія з добре налаштованими ознаками часто перевершує складні нейромережі без належної інфраструктури.""",
]

AI_MODERN_EN = [
    """The transition to cloud-native architectures rests on three interrelated principles — containerization, declarative configuration, and immutable infrastructure. Containerization ensures that application workloads are portable across environments, eliminating the classical "works on my machine" problem. Declarative configuration shifts the operational model from imperative deployment scripts to versioned state descriptions, which significantly improves reproducibility. Immutable infrastructure, in turn, treats running systems as disposable artifacts that are replaced rather than modified, which simplifies rollback and reduces configuration drift. Taken together, these principles form a coherent operational philosophy that has reshaped how modern software is delivered.""",

    """Effective product management requires balancing three competing pressures — customer needs, technical feasibility, and business viability. Customer needs provide the foundational orientation: any product disconnected from real user problems eventually struggles to find a sustainable market. Technical feasibility constrains what can be delivered within reasonable timeframes and resource budgets, and ignoring these constraints typically leads to ambitious roadmaps that fail in execution. Business viability, in turn, ensures that the product generates sufficient value to justify continued investment. Skilled product leaders navigate these three pressures simultaneously, recognizing that overoptimizing for any single dimension produces predictable failure modes.""",

    """The design of effective recommendation systems involves three core challenges — accurate preference modeling, diverse candidate generation, and appropriate ranking. Preference modeling builds a representation of user interests from observed behaviour and explicit feedback, which is fundamental for relevance. Candidate generation produces a manageable subset of items from potentially enormous catalogues, balancing personalization with discovery of new content. Ranking, the final stage, orders the candidates according to predicted utility, often incorporating signals beyond pure relevance — such as freshness, diversity, and business objectives. Modern recommendation pipelines typically address all three challenges in distinct architectural layers, allowing each to be optimized independently.""",

    """Organizational change management succeeds when three conditions align — clear vision, capable leadership, and sustained communication. A clear vision provides the destination: without it, change initiatives drift and lose momentum within months of launch. Capable leadership translates vision into action — identifying obstacles, marshalling resources, holding teams accountable for transitional milestones. Sustained communication ensures that the broader organization understands not only what is changing, but why, and remains engaged through the inevitable difficulties. When any of these three conditions is absent, even technically sound transformations frequently fail to deliver their expected benefits.""",

    """The architecture of a robust distributed system addresses three foundational concerns — fault tolerance, consistency, and scalability. Fault tolerance ensures that individual component failures do not cascade into systemic outages, which is achieved through redundancy, isolation, and graceful degradation. Consistency, defined formally through models such as linearizability or eventual consistency, governs how data appears across the system when concurrent operations occur. Scalability addresses how the system handles growth — in users, in data volume, in geographic footprint. These three concerns interact in non-trivial ways, as the CAP theorem famously demonstrates, and the design of any distributed system involves explicit tradeoffs among them.""",

    """The maturity of a software engineering organization is typically reflected in three dimensions — process discipline, technical excellence, and cultural alignment. Process discipline manifests in consistent practices for code review, testing, and deployment, ensuring predictable outcomes from a heterogeneous workforce. Technical excellence is observable in code quality metrics, architectural coherence, and the team's ability to evolve the system without accumulating debt at unsustainable rates. Cultural alignment, the least tangible of the three, determines whether teams collaborate effectively across boundaries — whether engineers feel safe to raise concerns, whether disagreements are resolved through evidence rather than authority. Mature organizations attend to all three simultaneously.""",

    """Successful machine learning deployments share three common characteristics — robust data pipelines, careful monitoring of model performance, and clear protocols for retraining. Robust data pipelines ensure that input distributions remain consistent and that schema changes do not silently degrade predictions. Careful monitoring extends beyond traditional service metrics to include drift detection, prediction quality, and segment-level performance analysis. Clear retraining protocols specify the triggers, validation steps, and rollback procedures for updating production models, which is essential because static models inevitably degrade as the world they describe evolves. Teams that neglect any of these areas typically encounter production incidents that could have been prevented.""",
]


# Англомовні «технічні» людські тексти — навмисно у тій самій тематиці,
# що Modern-LLM приклади, щоб калібратор навчився не плутати тему зі стилем.
HUMAN_EN_TECHNICAL = [
    """We tried migrating to microservices about three years ago. The pitch was compelling — independent deployments, technology flexibility, team autonomy. What actually happened was that our team of fifteen engineers ended up maintaining seventy-three services, most of which had no clear ownership. Two years later we've consolidated back to four services, and shipping velocity has roughly doubled. The lesson isn't that microservices are bad, it's that they require organizational maturity we didn't have at the time and probably still don't.""",

    """Our Prometheus setup grew organically over five years. By the end, we had alert rules that nobody could explain and dashboards that confused more than they helped. When the original engineer who built it left the company, the operational debt became impossible to ignore. We spent six months rewriting everything with stricter conventions, and I'd say maybe 60% of the rules survived the exercise. The rest were either redundant, broken, or alerting on things we no longer cared about. Monitoring is one of those areas where the cleanup costs more than the original implementation.""",

    """Switching from MongoDB to PostgreSQL on our main product was probably the most painful migration I've been involved in. We had four years of accumulated assumptions baked into the application code about document storage semantics. The eventual approach — running both databases in parallel for three months with dual writes — felt clumsy at the time but turned out to be the only safe path. Looking back, the original choice of MongoDB wasn't wrong for the early product, but we should have migrated two years earlier when the symptoms first appeared.""",

    """My team spent eight weeks on what was supposed to be a quick refactor of the authentication module. The original estimate was wrong because nobody on the current team had been around when the system was built. We kept finding undocumented edge cases — corporate SSO integrations from acquisitions, legacy mobile clients with custom token formats, internal tooling that used long-expired API keys. By the end we had touched basically every part of the codebase. The lesson, again: estimates for unfamiliar legacy code should be multiplied by three before being shared with anyone outside engineering.""",

    """Working on the database team at a large fintech taught me how much of distributed systems folklore is wrong. We ran into the famous CAP-theorem tradeoffs constantly, but rarely in the textbook way. Most of our real problems were operational — slow rollouts of schema changes, ambiguous semantics around schema evolution, runbooks that hadn't been updated for two years. The actual consistency model of our system wasn't strictly anything from the literature; it was whatever combination of partition tolerance and convergence behaviour our customers happened to tolerate.""",
]

# Modern-LLM огляди-порівняння — стиль критичного прикладу користувача
# (огляд інструментів моніторингу). Без цих прикладів калібратор не вчиться
# відрізняти такий «огляд» як ШІ-сигнатуру.
AI_MODERN_REVIEWS = [
    """Сучасний ринок CRM-систем представлений кількома основними рішеннями. Salesforce залишається лідером корпоративного сегмента завдяки розвиненій екосистемі інтеграцій та широким можливостям кастомізації, однак вартість впровадження робить його недоступним для невеликих команд. HubSpot пропонує більш доступний вхідний рівень з безкоштовною версією, проте функціональність обмежена в порівнянні з повноцінними корпоративними рішеннями. Pipedrive орієнтований переважно на відділи продажів і не передбачає глибокої інтеграції з маркетинговими процесами. Узагальнюючи, вибір CRM має базуватися не на популярності рішення, а на конкретних потребах команди та бюджеті проєкту.""",

    """Аналіз сучасних фреймворків для розробки веб-застосунків виявляє чітку диференціацію за призначенням. React залишається де-факто стандартом для динамічних інтерфейсів завдяки розвиненій екосистемі та широкому пулу спеціалістів, проте він не нав'язує архітектурних рішень, що може ускладнювати супровід великих проєктів. Vue.js пропонує більш виражену структуру з вбудованими механізмами управління станом, що робить його зручним для команд середнього розміру. Angular, своєю чергою, орієнтований на корпоративні рішення з суворою типізацією та централізованим управлінням залежностями. Вибір фреймворку повинен відповідати масштабу команди та довгостроковій стратегії підтримки проєкту.""",

    """Cucumber-тести у поєднанні з Selenium залишаються поширеним вибором для функціонального тестування веб-застосунків, особливо в командах, де важлива участь бізнес-аналітиків у формуванні тестових сценаріїв. Водночас підтримка Gherkin-сценаріїв вимагає істотних інженерних зусиль, які часто перевершують виграш від «читабельності» специфікацій. Playwright, на противагу цьому, надає сучаснішу архітектуру з вбудованою підтримкою паралельного виконання та автоматичним очікуванням елементів. Cypress зорієнтований на швидку зворотну зв'язок під час розробки, проте його обмеження у роботі з кросдоменними сценаріями стають критичними для складніших застосунків. Загалом, сучасні проєкти все частіше відмовляються від класичного BDD-підходу на користь інструментів з кращим розробницьким досвідом.""",

    """Контейнерні оркестратори представлені трьома основними категоріями рішень — повномасштабними платформами, легковажними дистрибутивами та керованими сервісами. Kubernetes у класичному вигляді залишається стандартом галузі та забезпечує максимальну гнучкість, проте операційна складність вимагає окремої команди з трьох-п'яти інженерів. Легковажні дистрибутиви на кшталт K3s або MicroK8s знижують поріг входу та підходять для edge-сценаріїв, але обмежені у функціональності для production-навантажень. Керовані сервіси — GKE, EKS, AKS — пропонують компроміс між контролем і операційними витратами, що робить їх оптимальним вибором для більшості команд середнього розміру. Підсумовуючи, вибір конкретного рішення має визначатися не модою, а реальними операційними можливостями організації.""",

    """Ринок інструментів управління проєктами насичений рішеннями різної зрілості та орієнтації. Jira залишається стандартом для команд, що працюють за методологією Scrum або Kanban, проте надлишкова гнучкість часто стає її проблемою — конфігурація проєкту може займати тижні. Linear надає сучасний інтерфейс і фокус на швидкості, що цінують команди розробки, але обмежений у можливостях звітності для менеджменту. Asana і Monday.com орієнтовані на ширшу аудиторію — від маркетингу до операційних команд — і програють у глибині інженерних функцій. Сукупно, вибір інструменту має враховувати як технічні потреби команди, так і її зв'язок з іншими підрозділами організації.""",

    """The contemporary landscape of monitoring and observability tools presents distinct tradeoffs across three main categories — open-source self-hosted stacks, managed SaaS platforms, and lightweight all-in-one solutions. Open-source stacks built around Prometheus and Grafana offer maximum flexibility and avoid vendor lock-in, but require dedicated engineering effort for initial deployment and ongoing maintenance. Managed SaaS platforms such as Datadog provide polished interfaces and broad integration coverage, yet impose significant subscription costs and constrain data sovereignty. Lightweight solutions like Netdata simplify single-node monitoring but typically lack the architectural sophistication needed for distributed environments. Overall, the optimal choice depends on team size, regulatory constraints, and the willingness to trade operational simplicity against long-term flexibility.""",

    """The evaluation of message-queue systems for production deployment involves balancing three principal dimensions — throughput characteristics, delivery semantics, and operational complexity. Apache Kafka remains the de-facto choice for high-throughput event streaming, providing strong durability guarantees but imposing significant operational overhead through its dependency on ZooKeeper or KRaft consensus. RabbitMQ offers more flexible routing patterns and simpler operations, yet its throughput ceiling limits applicability for high-volume scenarios. NATS provides exceptional simplicity and low latency, particularly with the JetStream extension, but its ecosystem of tooling remains less mature than competing options. Choosing among these systems should be driven by actual workload characteristics rather than by general reputation.""",
]


# --- Експорт у форматі датасету ---------------------------------------------

def build_calibration_dataset():
    """Повертає (texts, labels) для калібрування.

    Усі групи людських текстів отримують мітку 0, усі групи Modern-LLM
    текстів — мітку 1. Списки об'єднано в одному місці для зручності
    додавання нових прикладів у майбутньому.
    """
    human_groups = [
        HUMAN_UK_ACADEMIC,
        HUMAN_EN_ACADEMIC,
        HUMAN_UK_TECHNICAL,
        HUMAN_EN_TECHNICAL,
    ]
    ai_groups = [
        AI_MODERN_UK,
        AI_MODERN_EN,
        AI_MODERN_REVIEWS,
    ]

    texts = []
    labels = []
    for group in human_groups:
        for t in group:
            texts.append(t)
            labels.append(0)
    for group in ai_groups:
        for t in group:
            texts.append(t)
            labels.append(1)
    return texts, labels


if __name__ == "__main__":
    texts, labels = build_calibration_dataset()
    n_human = sum(1 for l in labels if l == 0)
    n_ai = sum(1 for l in labels if l == 1)
    print(f"Калібрувальний корпус: {len(texts)} текстів")
    print(f"  Людина (академічний+технічний, uk+en): {n_human}")
    print(f"  ШІ (Modern-LLM, uk+en):                {n_ai}")
