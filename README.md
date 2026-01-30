# FAS Smart - Real-Time Rolling Window Signal System

## Структура проекта

```
fas_smart/
├── .env                    # Конфигурация (не в git!)
├── .env.example            # Пример конфигурации
├── requirements.txt        # Python зависимости
├── sql/
│   ├── schema.sql          # Основная схема БД
│   └── migrations/         # Миграции
├── scripts/
│   ├── update_pairs.py     # Обновление списка пар
│   └── ...
└── src/
    ├── __init__.py
    ├── config.py           # Загрузка конфигурации
    ├── db.py               # Подключение к БД
    └── ...
```

## Установка

```bash
# Клонировать/скопировать проект
cd ~/fas_smart

# Создать venv
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Настроить .env
cp .env.example .env
nano .env
```

## Запуск

```bash
# Обновить пары
python scripts/update_pairs.py
```
