# Revert LP Strategy

Статистическая система анализа и мониторинга LP-позиций для Revert Finance и Uniswap V3.

## Возможности

### Модуль 1: Анализ новых токенов
- Мониторинг недавно созданных пулов
- Анализ LP-позиций на новых пулах
- Рекомендации по диапазонам на основе статистики

### Модуль 2: Потоки vs Цена
- Корреляция между объёмом торгов и изменением цены
- Расчёт порогов для значимых потоков
- Сигналы при достижении порогов

### Модуль 3: Детекция сливов капитала
- Мониторинг крупных оттоков из пулов
- Абсолютные и относительные пороги
- Уведомления через Telegram

### Модуль 4: Анализ владельцев LP
- Расчёт PnL по каждому владельцу позиций
- Рейтинг успешных LP-провайдеров
- Извлечение паттернов успешных стратегий

## Поддерживаемые сети

- Ethereum
- Arbitrum
- Polygon
- Optimism
- Base
- BNB Chain
- Unichain (скоро)

## Установка

```bash
# Клонирование (если ещё не сделано)
cd /home/gevorg/Downloads/webi/revert-lp-strategy

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Копирование конфига
cp .env.example .env
```

### Настройка The Graph API Key (обязательно)

Проект использует The Graph Decentralized Network для получения данных о пулах и свопах.
Для этого **необходим бесплатный API key**:

1. Зайдите на [The Graph Studio](https://thegraph.com/studio/)
2. Создайте аккаунт (через GitHub или email)
3. В разделе "API Keys" создайте новый ключ
4. Скопируйте ключ в `.env`:

```bash
GRAPH_API_KEY=your_api_key_here
```

**Бесплатный лимит:** 100,000 запросов в месяц — достаточно для разработки и тестирования.

## Быстрый старт

### 1. Инициализация базы данных

```bash
python scripts/init_db.py
```

### 2. Загрузка пулов

```bash
# Все сети
python scripts/fetch_pools.py

# Только определённые сети
python scripts/fetch_pools.py --networks ethereum,arbitrum

# С кастомным минимальным TVL
python scripts/fetch_pools.py --min-tvl 100000
```

### 3. Анализ оттоков капитала

```bash
# Загрузить свопы и запустить анализ
python scripts/run_analysis.py --load-swaps --hours 1

# С уведомлениями в Telegram
python scripts/run_analysis.py --load-swaps --notify
```

### 4. Анализ владельцев LP

```bash
# Топ-20 владельцев по PnL
python scripts/analyze_owners.py --top 20

# Паттерны успешных LP
python scripts/analyze_owners.py --patterns

# Сохранить статистику в БД
python scripts/analyze_owners.py --save
```

## Структура проекта

```
revert-lp-strategy/
├── config/
│   ├── __init__.py
│   ├── settings.py      # Настройки и пороги
│   └── networks.py      # Конфигурация сетей
├── src/
│   ├── data/
│   │   ├── subgraph.py  # Клиент для The Graph
│   │   ├── pools.py     # ETL пулов
│   │   └── swaps.py     # ETL свопов
│   ├── analytics/
│   │   ├── capital_flow.py  # Модуль 3: сливы
│   │   └── owners.py        # Модуль 4: владельцы
│   ├── signals/
│   │   └── telegram.py      # Уведомления
│   ├── db/
│   │   ├── database.py      # Подключение к БД
│   │   └── models.py        # SQLAlchemy модели
│   └── utils/
│       └── helpers.py       # Вспомогательные функции
├── scripts/
│   ├── init_db.py           # Инициализация БД
│   ├── fetch_pools.py       # Загрузка пулов
│   ├── run_analysis.py      # Запуск анализа
│   └── analyze_owners.py    # Анализ владельцев
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Конфигурация

### Переменные окружения (.env)

```bash
# База данных
DATABASE_URL=sqlite:///./data/revert_lp.db

# Telegram уведомления
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Режим отладки
DEBUG=false
```

### Настройки порогов (config/settings.py)

Основные пороги можно менять в `config/settings.py`:

- `CapitalFlowSettings.large_outflow_usd` — порог крупного оттока в USD
- `CapitalFlowSettings.large_outflow_tvl_percent` — порог в % от TVL
- `OwnerAnalysisSettings.min_positions` — минимум позиций для статистики
- `PoolFilterSettings.min_tvl_usd` — минимальный TVL пула

## Telegram Bot

Для получения уведомлений:

1. Создайте бота через [@BotFather](https://t.me/botfather)
2. Получите токен бота
3. Напишите боту любое сообщение
4. Получите chat_id через API: `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Добавьте токен и chat_id в `.env`

## Следующие шаги

- [ ] Модуль 1: Анализ новых токенов
- [ ] Модуль 2: Корреляция потоков и цены
- [ ] Backtesting framework
- [ ] Scheduler для автоматического запуска
- [ ] Web UI для мониторинга

## Лицензия

MIT
