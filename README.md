Итоговый проект курса "Машинное обучение в бизнесе"

Стек:
ML pandas, numpy, dill, flask, sklearn

Данные из задания первого урока курса. Находятся в директории проекта for_jupyter/WA_Fn-UseC_-Telco-Customer-Churn

Задача: (первого урока курса) предсказание по описаниям использования телевизионных сервисов, продолжения взаимодействия. Бинарная классификация

Используемые признаки:
1. gender - пол (0, 1) (женский, мужской)
2. tenure - устройств во владении, тип - целое число
3. PhoneService - телефонное обслуживание (0, 1)
4. TotalCharges - суммарная стоимость, тип число
5. StreamingMovies - просмотр фильмлв (0, 1)
6. StreamingTV - просмотр ТВ (0, 1)
7. TechSupport - техническая поддержка (0, 1)

Целевая переменная: Churn

Модель: model_curs

В директории for_python находятся три ноутбука. 
1. Step1_api - сборка пайплайна. 
2. Step2_api - проверка работоспособности пайплайна. 
3. Step3_api - получение предсказаний, после запуска файла run_server
4. Файл с данными используемыми при обучении модели WA_Fn-UseC_-Telco-Customer-Churn
5. Файлы X_test и y_test, используемые для проверки работоспособности пайплайна

Клонируем репозиторий:
git clone https://github.com/AlekseyZhilin/api_curs_project

Linux у меня нет, поэтому docker сделать не могу. Для использования, предлагаю следующий вариант (только api):

При необходимости, устанавливаем библиотеки pip install -r requirements.txt
1. Переносим директорию app в каталог пользователя
2. В коммандной строке пишем: сd app
3. В коммандной строке пишем: python run_server.py

Запускаем контейнер

Переходим на http://127.0.0.1:5000

После всех выполненных действий, можно запустить ноутбук Step3_api и получить предсказания. По ссллке http://127.0.0.1:5000, можно не переходить.
