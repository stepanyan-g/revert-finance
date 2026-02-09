"""
Revert LP Strategy - Web Dashboard (Русская версия)

Запуск: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

# Import project modules
from config.networks import NETWORKS, GRAPH_API_KEY
from config.settings import get_settings
from src.db.database import init_db, session_scope
from src.db.models import Pool, Token, Swap, Signal, Position, Owner, WatchedOwner
from src.data.pools import PoolLoader
from src.data.swaps import SwapLoader, SwapAnalyzer
from src.data.positions import PositionLoader
from src.analytics.capital_flow import CapitalFlowAnalyzer, detect_large_outflows
from src.analytics.new_tokens import NewTokenAnalyzer, get_new_pools
from src.analytics.flow_price import FlowPriceAnalyzer, analyze_token_flows
from src.analytics.owners import OwnerAnalyzer, get_top_lp_owners

# Page config
st.set_page_config(
    page_title="Revert LP Стратегия",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()


# =============================================================================
# Helper Functions
# =============================================================================

def get_pool_stats() -> dict:
    """Получить статистику пулов из базы данных."""
    with session_scope() as session:
        from sqlalchemy import func
        
        total_pools = session.query(Pool).count()
        total_tokens = session.query(Token).count()
        total_swaps = session.query(Swap).count()
        total_positions = session.query(Position).count()
        total_owners = session.query(Owner).count()
        
        pools_by_network = dict(
            session.query(Pool.network, func.count(Pool.id))
            .group_by(Pool.network)
            .all()
        )
        
        return {
            "total_pools": total_pools,
            "total_tokens": total_tokens,
            "total_swaps": total_swaps,
            "total_positions": total_positions,
            "total_owners": total_owners,
            "pools_by_network": pools_by_network,
        }


def get_top_pools(limit: int = 20, network: str = None) -> pd.DataFrame:
    """Получить топ пулов по TVL."""
    with session_scope() as session:
        query = session.query(Pool).filter(Pool.tvl_usd < 10_000_000_000)
        
        if network and network != "Все":
            query = query.filter(Pool.network == network)
        
        pools = query.order_by(Pool.tvl_usd.desc()).limit(limit).all()
        
        data = []
        for p in pools:
            data.append({
                "Пул": f"{p.token0_symbol}/{p.token1_symbol}",
                "Сеть": p.network,
                "TVL ($)": float(p.tvl_usd or 0),
                "Комиссия": f"{(p.fee_tier or 0) / 10000:.2f}%",
                "Адрес": p.address[:10] + "...",
            })
        
        return pd.DataFrame(data)


def get_recent_swaps(limit: int = 50, network: str = None) -> pd.DataFrame:
    """Получить недавние свопы."""
    with session_scope() as session:
        query = session.query(Swap, Pool).join(Pool)
        
        if network and network != "Все":
            query = query.filter(Swap.network == network)
        
        swaps = query.order_by(Swap.timestamp.desc()).limit(limit).all()
        
        data = []
        for swap, pool in swaps:
            data.append({
                "Время": swap.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Пул": f"{pool.token0_symbol}/{pool.token1_symbol}",
                "Сеть": swap.network,
                "Сумма ($)": float(swap.amount_usd or 0),
                "Тип": "Покупка" if swap.direction == "buy" else "Продажа",
            })
        
        return pd.DataFrame(data)


def get_signals(limit: int = 20) -> pd.DataFrame:
    """Получить сигналы."""
    with session_scope() as session:
        signals = session.query(Signal).order_by(Signal.created_at.desc()).limit(limit).all()
        
        data = []
        for s in signals:
            severity_ru = {"critical": "Критический", "warning": "Внимание", "info": "Инфо"}
            data.append({
                "Время": s.created_at.strftime("%Y-%m-%d %H:%M"),
                "Тип": s.signal_type,
                "Важность": severity_ru.get(s.severity, s.severity),
                "Название": s.title,
                "Сумма ($)": float(s.amount_usd or 0),
                "Сеть": s.network or "-",
                "Отправлен": "✅" if s.is_sent else "❌",
            })
        
        return pd.DataFrame(data)


def load_all_data_action(networks: list, min_tvl: float, positions_limit: int):
    """Загрузить все данные: пулы → свопы → позиции."""
    results = {
        "pools": {},
        "swaps": 0,
        "positions": {},
    }
    
    progress = st.progress(0)
    status = st.empty()
    
    # Шаг 1: Загрузка пулов
    status.text("📊 Шаг 1/4: Загрузка пулов...")
    loader = PoolLoader()
    
    for i, network in enumerate(networks):
        try:
            with session_scope() as session:
                count = loader.load_pools_for_network(session, network, min_tvl=min_tvl)
                results["pools"][network] = count
        except Exception as e:
            results["pools"][network] = f"Ошибка: {str(e)[:30]}"
        progress.progress((i + 1) / len(networks) * 0.25)
    
    # Шаг 2: Загрузка свопов
    status.text("💱 Шаг 2/4: Загрузка свопов...")
    with session_scope() as session:
        pools = session.query(Pool).filter(
            Pool.tvl_usd >= min_tvl,
            Pool.network.in_(networks)
        ).order_by(Pool.tvl_usd.desc()).limit(30).all()
        
        swap_loader = SwapLoader()
        for i, pool in enumerate(pools):
            try:
                count = swap_loader.load_swaps_for_pool(session, pool, limit=50)
                results["swaps"] += count
            except:
                pass
            progress.progress(0.25 + (i + 1) / len(pools) * 0.25)
    
    # Шаг 3: Загрузка позиций (открытые + закрытые через mints/burns)
    status.text("📍 Шаг 3/4: Загрузка позиций через события mint/burn...")
    pos_loader = PositionLoader()
    
    for i, network in enumerate(networks):
        try:
            with session_scope() as session:
                # Загружает открытые И закрытые позиции через анализ mint/burn событий
                result = pos_loader.load_positions_from_events(
                    session, network, min_amount_usd="100", limit=positions_limit
                )
                results["positions"][network] = result
        except Exception as e:
            results["positions"][network] = {"open": 0, "closed": 0, "error": str(e)[:30]}
        progress.progress(0.50 + (i + 1) / len(networks) * 0.35)
    
    # Шаг 4: Расчёт USD для позиций
    status.text("💵 Шаг 4/4: Расчёт USD-значений для позиций...")
    calculate_positions_usd()
    
    progress.progress(1.0)
    status.text("✅ Загрузка завершена!")
    
    return results


def calculate_positions_usd():
    """Рассчитать USD-значения для позиций на основе цен пулов."""
    with session_scope() as session:
        # Получаем все позиции без USD
        positions = session.query(Position).filter(
            Position.deposited_usd.is_(None)
        ).all()
        
        for pos in positions:
            # Получаем пул для цен
            pool = session.query(Pool).filter(Pool.id == pos.pool_id).first()
            if not pool:
                continue
            
            # Используем текущие цены пула (упрощённо)
            # В реальности нужны исторические цены
            price0 = float(pool.token0_price or 0)
            price1 = float(pool.token1_price or 0)
            
            # Если цен нет, пробуем оценить через TVL
            if price0 == 0 and price1 == 0:
                # Грубая оценка: если один из токенов стейблкоин
                if pool.token0_symbol in ("USDC", "USDT", "DAI", "BUSD"):
                    price0 = 1.0
                if pool.token1_symbol in ("USDC", "USDT", "DAI", "BUSD"):
                    price1 = 1.0
                if pool.token0_symbol in ("WETH", "ETH"):
                    price0 = 3000.0  # Примерная цена ETH
                if pool.token1_symbol in ("WETH", "ETH"):
                    price1 = 3000.0
                if pool.token0_symbol in ("WBTC", "BTC"):
                    price0 = 100000.0  # Примерная цена BTC
                if pool.token1_symbol in ("WBTC", "BTC"):
                    price1 = 100000.0
            
            # Получаем decimals (по умолчанию 18)
            decimals0 = 18
            decimals1 = 18
            
            # Конвертируем в USD
            dep0 = float(pos.deposited_token0 or 0) / (10 ** decimals0)
            dep1 = float(pos.deposited_token1 or 0) / (10 ** decimals1)
            with0 = float(pos.withdrawn_token0 or 0) / (10 ** decimals0)
            with1 = float(pos.withdrawn_token1 or 0) / (10 ** decimals1)
            fees0 = float(pos.collected_fees_token0 or 0) / (10 ** decimals0)
            fees1 = float(pos.collected_fees_token1 or 0) / (10 ** decimals1)
            
            pos.deposited_usd = Decimal(str(dep0 * price0 + dep1 * price1))
            pos.withdrawn_usd = Decimal(str(with0 * price0 + with1 * price1))
            pos.collected_fees_usd = Decimal(str(fees0 * price0 + fees1 * price1))
            
            # Рассчитываем текущую стоимость (упрощённо)
            if not pos.is_closed:
                # Текущая стоимость ≈ депозит (без IL расчёта)
                pos.current_value_usd = pos.deposited_usd
        
        session.commit()


def run_analysis_action(networks: list, hours: int):
    """Запустить анализ оттоков."""
    net_filter = networks if networks and "Все" not in networks else None
    
    with st.spinner("Анализ оттоков капитала..."):
        alerts = detect_large_outflows(
            hours=hours,
            networks=net_filter,
            save_signals=True,
        )
    
    return alerts


def get_owner_positions(owner_address: str) -> pd.DataFrame:
    """Получить все позиции владельца."""
    with session_scope() as session:
        positions = session.query(Position, Pool).outerjoin(Pool).filter(
            Position.owner_address == owner_address.lower()
        ).order_by(Position.created_at.desc()).all()
        
        data = []
        for pos, pool in positions:
            pool_name = f"{pool.token0_symbol}/{pool.token1_symbol}" if pool else pos.pool_address[:10]
            
            # Платформа (DEX)
            dex_name = pos.dex.upper().replace("_", " ") if pos.dex else "Uniswap V3"
            
            # Адрес LP пула
            pool_addr = pos.pool_address if pos.pool_address else "-"
            
            # Рассчитаем PnL
            dep = float(pos.deposited_usd or 0)
            wit = float(pos.withdrawn_usd or 0)
            fees = float(pos.collected_fees_usd or 0)
            pnl = wit + fees - dep if pos.is_closed else 0
            
            data.append({
                "Платформа": dex_name,
                "Сеть": pos.network,
                "Пул": pool_name,
                "Адрес LP": pool_addr,
                "Статус": "🔒 Закрыта" if pos.is_closed else "🟢 Открыта",
                "Диапазон": f"{pos.tick_lower} → {pos.tick_upper}",
                "Депозит ($)": round(dep, 2),
                "Вывод ($)": round(wit, 2),
                "Комиссии ($)": round(fees, 2),
                "PnL ($)": round(pnl, 2),
                "Открыта": pos.created_at.strftime("%Y-%m-%d %H:%M") if pos.created_at else "-",
                "Закрыта": pos.closed_at.strftime("%Y-%m-%d %H:%M") if pos.closed_at else "-",
            })
        
        return pd.DataFrame(data)


def add_to_watchlist(owner_address: str, note: str = "") -> str:
    """
    Добавить владельца в список отслеживания.
    
    Returns:
        "success" - успешно добавлен
        "exists" - уже в списке
        "not_found" - владелец не найден в базе
        "error" - ошибка
    """
    try:
        with session_scope() as session:
            # Проверяем существует ли
            existing = session.query(WatchedOwner).filter(
                WatchedOwner.owner_address == owner_address.lower()
            ).first()
            
            if existing:
                return "exists"
            
            # Находим owner_id
            owner = session.query(Owner).filter(
                Owner.address == owner_address.lower()
            ).first()
            
            if not owner:
                # Попробуем создать Owner если его нет но есть позиции
                pos_count = session.query(Position).filter(
                    Position.owner_address == owner_address.lower()
                ).count()
                
                if pos_count == 0:
                    return "not_found"
                
                # Создаём Owner
                owner = Owner(
                    address=owner_address.lower(),
                    is_contract=owner_address.lower().startswith("0x000000"),
                )
                session.add(owner)
                session.flush()
            
            # Считаем текущие позиции
            pos_count = session.query(Position).filter(
                Position.owner_address == owner_address.lower()
            ).count()
            
            watched = WatchedOwner(
                owner_id=owner.id,
                owner_address=owner_address.lower(),
                note=note,
                last_position_count=pos_count,
                last_checked_at=datetime.now(),
            )
            session.add(watched)
            session.commit()
            return "success"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "error"


def remove_from_watchlist(owner_address: str) -> bool:
    """Удалить владельца из списка отслеживания."""
    with session_scope() as session:
        watched = session.query(WatchedOwner).filter(
            WatchedOwner.owner_address == owner_address.lower()
        ).first()
        
        if watched:
            session.delete(watched)
            session.commit()
            return True
        return False


def get_watched_owners() -> list:
    """Получить список отслеживаемых владельцев."""
    with session_scope() as session:
        watched = session.query(WatchedOwner).all()
        
        result = []
        for w in watched:
            # Получаем актуальное количество позиций
            current_count = session.query(Position).filter(
                Position.owner_address == w.owner_address
            ).count()
            
            open_count = session.query(Position).filter(
                Position.owner_address == w.owner_address,
                Position.is_closed == False
            ).count()
            
            closed_count = session.query(Position).filter(
                Position.owner_address == w.owner_address,
                Position.is_closed == True
            ).count()
            
            result.append({
                "address": w.owner_address,
                "note": w.note or "",
                "added_at": w.added_at,
                "last_checked": w.last_checked_at,
                "last_count": w.last_position_count,
                "current_count": current_count,
                "open_positions": open_count,
                "closed_positions": closed_count,
                "new_activity": current_count != w.last_position_count,
                "notify_telegram": w.notify_telegram,
            })
        
        return result


# =============================================================================
# Sidebar (Русское меню)
# =============================================================================

st.sidebar.title("🔄 Revert LP Стратегия")
st.sidebar.markdown("---")

# API Key status
if GRAPH_API_KEY:
    st.sidebar.success("✅ API ключ настроен")
else:
    st.sidebar.error("❌ GRAPH_API_KEY не установлен")

st.sidebar.markdown("---")

# Navigation with descriptions
page = st.sidebar.radio(
    "📌 Навигация",
    [
        "🏠 Главная",
        "📥 Загрузка данных",
        "🔍 Анализ оттоков",
        "💰 Потоки токенов",
        "🆕 Новые пулы",
        "🏆 Топ владельцев LP",
        "👁️ Мониторинг",
        "⚠️ Сигналы",
        "⚙️ Настройки",
    ],
)

# Tooltips in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Справка")

with st.sidebar.expander("Как пользоваться?"):
    st.markdown("""
    **1. Загрузка данных**
    - Нажмите «Загрузить всё» на странице загрузки
    - Данные сохраняются в базу и доступны при следующем запуске
    
    **2. Топ владельцев LP** ⭐
    - Рейтинг по **% успешно закрытых позиций**
    - Нажмите на владельца — увидите ВСЕ его позиции
    - Кнопка «Добавить в мониторинг» для отслеживания
    
    **3. Мониторинг** 👁️
    - Следите за успешными LP в реальном времени
    - Telegram-уведомления при новой активности
    - Добавляйте владельцев из «Топ LP»
    
    **4. Сигналы**
    - История всех обнаруженных событий
    - Можно настроить Telegram-уведомления
    """)


# =============================================================================
# Главная страница
# =============================================================================

if page == "🏠 Главная":
    st.title("📊 Главная панель")
    
    st.info("""
    **Добро пожаловать в Revert LP Strategy!**
    
    Эта система анализирует LP-позиции на Uniswap V3 и помогает:
    - 🔍 Находить крупные оттоки капитала (сливы)
    - 💰 Отслеживать потоки по токенам
    - 🏆 Изучать успешных LP-провайдеров
    - 🆕 Мониторить новые пулы
    """)
    
    # Stats
    stats = get_pool_stats()
    
    st.markdown("### 📈 Статистика базы данных")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Пулы", f"{stats['total_pools']:,}")
    col2.metric("Токены", f"{stats['total_tokens']:,}")
    col3.metric("Свопы", f"{stats['total_swaps']:,}")
    col4.metric("Позиции", f"{stats['total_positions']:,}")
    col5.metric("Владельцы", f"{stats['total_owners']:,}")
    
    if stats['total_pools'] == 0:
        st.warning("⚠️ Данные не загружены. Перейдите в раздел «📥 Загрузка данных».")
    else:
        st.markdown("---")
        
        # Pools by network
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌐 Пулы по сетям")
            if stats['pools_by_network']:
                df = pd.DataFrame([
                    {"Сеть": k, "Пулы": v} 
                    for k, v in stats['pools_by_network'].items()
                ])
                st.bar_chart(df.set_index("Сеть"))
        
        with col2:
            st.markdown("### 🏊 Топ пулов по TVL")
            network_options = ["Все"] + list(stats['pools_by_network'].keys())
            network_filter = st.selectbox("Фильтр по сети", network_options)
            df = get_top_pools(limit=10, network=network_filter if network_filter != "Все" else None)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Recent swaps
        st.markdown("### 💱 Последние свопы")
        df_swaps = get_recent_swaps(limit=15)
        if not df_swaps.empty:
            st.dataframe(df_swaps, use_container_width=True, hide_index=True)
        else:
            st.info("Свопы не загружены")


# =============================================================================
# Загрузка данных
# =============================================================================

elif page == "📥 Загрузка данных":
    st.title("📥 Загрузка данных")
    
    # Import period loader
    from src.data.period_loader import (
        PeriodDataLoader, get_period_options, get_multi_period_options
    )
    
    st.info("""
    **Как это работает:**
    - Выберите период (месяц или несколько месяцев) для загрузки
    - Таблица показывает сколько данных доступно и сколько уже загружено
    - Данные загружаются инкрементально (новые добавляются, существующие не дублируются)
    """)
    
    # Current stats
    stats = get_pool_stats()
    
    st.markdown("### 📊 Текущие данные в базе")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Пулы", stats['total_pools'])
    col2.metric("Свопы", stats['total_swaps'])
    col3.metric("Позиции", stats['total_positions'])
    col4.metric("Владельцы", stats['total_owners'])
    
    st.markdown("---")
    
    # =============================================================================
    # Filters
    # =============================================================================
    st.markdown("### ⚙️ Параметры загрузки")
    
    available_networks = [n for n, c in NETWORKS.items() if c.enabled]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Period selection
        period_options = get_multi_period_options()
        period_labels = [label for label, _ in period_options]
        
        selected_period_label = st.selectbox(
            "📅 Период",
            period_labels,
            index=0,
            help="Выберите период для загрузки данных"
        )
        
        # Get the actual periods for selected label
        selected_periods = None
        for label, periods in period_options:
            if label == selected_period_label:
                selected_periods = periods
                break
    
    with col2:
        selected_networks = st.multiselect(
            "🌐 Сети",
            available_networks,
            default=["arbitrum", "ethereum"] if "arbitrum" in available_networks else available_networks[:2],
            help="Выберите сети для загрузки"
        )
    
    with col3:
        min_tvl = st.number_input(
            "💰 Мин. TVL пула ($)",
            min_value=10000,
            value=100000,
            step=10000,
            help="Пулы с TVL меньше этого значения не загружаются"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        min_amount = st.number_input(
            "💵 Мин. сумма позиции ($)",
            min_value=10,
            value=100,
            step=50,
            help="Позиции меньше этой суммы не загружаются"
        )
    with col2:
        limit_per_period = st.number_input(
            "📊 Лимит на период",
            min_value=100,
            value=500,
            step=100,
            help="Максимум записей за один период"
        )
    
    st.markdown("---")
    
    # =============================================================================
    # Statistics Table
    # =============================================================================
    st.markdown("### 📈 Статистика загрузки")
    
    if selected_periods and selected_networks:
        # Show loading statistics
        if st.button("🔄 Обновить статистику", key="refresh_stats"):
            with st.spinner("Получение статистики..."):
                with session_scope() as session:
                    loader = PeriodDataLoader()
                    loader.refresh_statistics(
                        session, selected_networks, min_tvl, min_amount
                    )
            st.rerun()
        
        with session_scope() as session:
            loader = PeriodDataLoader()
            
            # Get stats for positions
            position_stats = loader.get_period_statistics(
                session, selected_periods, selected_networks,
                min_tvl_usd=min_tvl,
                min_amount_usd=min_amount,
                data_type="positions"
            )
            
            # Get stats for pools
            pool_stats = loader.get_period_statistics(
                session, selected_periods, selected_networks,
                min_tvl_usd=min_tvl,
                min_amount_usd=min_amount,
                data_type="pools"
            )
        
        # Create DataFrame for display
        if position_stats:
            st.markdown("#### 📍 Позиции")
            
            pos_data = []
            for s in position_stats:
                status_icon = "✅" if s.is_fully_loaded else ("🔶" if s.loaded_percent > 0 else "⬜")
                pos_data.append({
                    "Статус": status_icon,
                    "Период": s.period_label,
                    "Сеть": s.network,
                    "В блокчейне": s.total_available,
                    "Загружено": s.total_loaded,
                    "Прогресс": f"{s.loaded_percent:.0f}%",
                })
            
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            
            # Summary
            total_available = sum(s.total_available for s in position_stats)
            total_loaded = sum(s.total_loaded for s in position_stats)
            overall_percent = (total_loaded / total_available * 100) if total_available > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Всего в блокчейне", f"{total_available:,}")
            col2.metric("Загружено", f"{total_loaded:,}")
            col3.metric("Общий прогресс", f"{overall_percent:.1f}%")
        
        if pool_stats:
            st.markdown("#### 🏊 Пулы")
            
            pool_data = []
            for s in pool_stats:
                status_icon = "✅" if s.is_fully_loaded else ("🔶" if s.loaded_percent > 0 else "⬜")
                pool_data.append({
                    "Статус": status_icon,
                    "Сеть": s.network,
                    "В блокчейне": s.total_available,
                    "Загружено": s.total_loaded,
                    "Прогресс": f"{s.loaded_percent:.0f}%",
                })
            
            pool_df = pd.DataFrame(pool_data)
            st.dataframe(pool_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Выберите период и хотя бы одну сеть для просмотра статистики")
    
    st.markdown("---")
    
    # =============================================================================
    # Load Data Button
    # =============================================================================
    st.markdown("### 🚀 Загрузка данных")
    
    if st.button("🚀 Загрузить данные за выбранный период", type="primary", use_container_width=True):
        if not selected_networks:
            st.error("Выберите хотя бы одну сеть")
        elif not GRAPH_API_KEY:
            st.error("GRAPH_API_KEY не настроен. Добавьте его в файл .env")
        elif not selected_periods:
            st.error("Выберите период")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)
            
            with session_scope() as session:
                loader = PeriodDataLoader()
                results = loader.load_period_data(
                    session,
                    selected_periods,
                    selected_networks,
                    min_tvl_usd=min_tvl,
                    min_amount_usd=min_amount,
                    limit_per_period=limit_per_period,
                    progress_callback=update_progress,
                )
            
            # Calculate positions USD
            calculate_positions_usd()
            
            st.success("✅ Загрузка завершена!")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Пулы:**")
                total_pools = 0
                for key, count in results["pools"].items():
                    st.write(f"• {key}: {count}")
                    total_pools += count
                st.write(f"**Итого: {total_pools}**")
            
            with col2:
                st.markdown("**Позиции:**")
                total_pos = 0
                for key, data in results["positions"].items():
                    if isinstance(data, dict):
                        count = data.get('open', 0) + data.get('closed', 0)
                        st.write(f"• {key}: {count} (откр: {data.get('open', 0)}, закр: {data.get('closed', 0)})")
                        total_pos += count
                    else:
                        st.write(f"• {key}: {data}")
                st.write(f"**Итого: {total_pos}**")
            
            if results["errors"]:
                with st.expander("⚠️ Ошибки при загрузке"):
                    for err in results["errors"]:
                        st.write(f"• {err}")
            
            st.rerun()
    
    # =============================================================================
    # Manual loading (legacy)
    # =============================================================================
    with st.expander("⚙️ Ручная загрузка (для продвинутых)"):
        st.markdown("""
        **Примечание:** Рекомендуется использовать загрузку по периодам выше.
        Эти опции сохранены для обратной совместимости.
        """)
        
        tab1, tab2 = st.tabs(["Пулы", "Позиции"])
        
        with tab1:
            st.markdown("Загрузить только пулы:")
            if st.button("Загрузить пулы", key="manual_load_pools"):
                loader = PoolLoader()
                with st.spinner("Загрузка пулов..."):
                    with session_scope() as session:
                        for net in selected_networks:
                            loader.load_pools_for_network(session, net, min_tvl=min_tvl)
                st.success("Пулы загружены!")
        
        with tab2:
            st.markdown("Загрузить только позиции:")
            if st.button("Загрузить позиции", key="manual_load_positions"):
                pos_loader = PositionLoader()
                with st.spinner("Загрузка позиций..."):
                    results = pos_loader.load_all_positions(
                        networks=selected_networks,
                        limit_per_network=limit_per_period
                    )
                    calculate_positions_usd()
                st.success(f"Позиции загружены: {results}")


# =============================================================================
# Анализ оттоков
# =============================================================================

elif page == "🔍 Анализ оттоков":
    st.title("🔍 Анализ оттоков капитала")
    
    st.info("""
    **Что это:**
    Система находит пулы, где за последнее время был **крупный чистый отток** 
    (продажи > покупки). Это может сигнализировать о «сливе» токена.
    
    **Как читать:**
    - 🔴 Критический — очень большой отток (> 5% TVL)
    - 🟠 Внимание — значительный отток
    - 🔵 Инфо — заметный отток
    """)
    
    stats = get_pool_stats()
    
    if stats['total_swaps'] == 0:
        st.warning("⚠️ Нет данных о свопах. Сначала загрузите данные.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            networks_list = ["Все"] + list(stats['pools_by_network'].keys())
            selected_networks = st.multiselect(
                "Сети",
                networks_list,
                default=["Все"],
                help="Выберите сети для анализа"
            )
        
        with col2:
            hours = st.slider(
                "Период анализа (часы)", 
                1, 72, 24,
                help="За какой период искать оттоки"
            )
        
        if st.button("🔍 Найти оттоки", type="primary"):
            alerts = run_analysis_action(selected_networks, hours)
            
            if alerts:
                st.success(f"Найдено {len(alerts)} событий оттока!")
                
                for alert in alerts:
                    severity_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
                    icon = severity_icon.get(alert.severity, "⚪")
                    
                    with st.expander(f"{icon} {alert.token0_symbol}/{alert.token1_symbol} ({alert.network})"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Чистый отток", f"${abs(alert.net_flow_usd):,.0f}")
                        col2.metric("TVL пула", f"${alert.tvl_usd:,.0f}")
                        col3.metric("% от TVL", f"{alert.outflow_percent_of_tvl:.1f}%")
                        
                        st.write(f"**Свопов:** {alert.swap_count}")
                        st.write(f"**Крупнейший своп:** ${alert.largest_swap_usd:,.0f}")
            else:
                st.info("✅ Значительных оттоков не обнаружено")
        
        st.markdown("---")
        
        # Token flow summary
        st.markdown("### 📊 Сводка потоков по токенам")
        
        with session_scope() as session:
            analyzer = SwapAnalyzer()
            flow_data = analyzer.get_flow_by_token(session, hours=24, min_volume_usd=10000)
            
            if flow_data:
                df = pd.DataFrame(flow_data)
                df = df.rename(columns={
                    "token_symbol": "Токен",
                    "network": "Сеть",
                    "inflow_usd": "Приток ($)",
                    "outflow_usd": "Отток ($)",
                    "net_flow_usd": "Чистый поток ($)",
                    "swap_count": "Свопов",
                })
                df = df.sort_values("Чистый поток ($)", key=abs, ascending=False).head(20)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Нет данных о потоках")


# =============================================================================
# Потоки токенов
# =============================================================================

elif page == "💰 Потоки токенов":
    st.title("💰 Анализ потоков токенов")
    
    st.info("""
    **Что это:**
    Показывает **чистый поток** по каждому токену (приток минус отток).
    
    **Интерпретация:**
    - 📈 Положительный поток = больше покупок (бычий сигнал)
    - 📉 Отрицательный поток = больше продаж (медвежий сигнал)
    - Percentile показывает насколько текущий поток необычен (>90% = очень необычно)
    """)
    
    stats = get_pool_stats()
    
    if stats['total_swaps'] == 0:
        st.warning("⚠️ Нет данных. Загрузите свопы на странице «Загрузка данных».")
    else:
        col1, col2 = st.columns(2)
        with col1:
            hours = st.slider("Период (часы)", 1, 168, 24, key="flow_hours")
        with col2:
            min_volume = st.number_input("Мин. объём ($)", value=10000, step=1000)
        
        if st.button("📊 Анализировать потоки", type="primary"):
            flows = analyze_token_flows(hours=hours)
            
            if flows:
                flows = [f for f in flows if (f.inflow_usd + f.outflow_usd) >= min_volume]
                
                st.success(f"Найдено {len(flows)} токенов")
                
                data = []
                for f in flows[:50]:
                    direction = "📈" if f.net_flow_usd > 0 else "📉"
                    data.append({
                        "": direction,
                        "Токен": f.token_symbol,
                        "Сеть": f.network,
                        "Чистый поток ($)": f.net_flow_usd,
                        "Приток ($)": f.inflow_usd,
                        "Отток ($)": f.outflow_usd,
                        "Свопов": f.swap_count,
                        "Percentile": f"{f.flow_percentile:.0f}%",
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Highlight significant
                significant = [f for f in flows if f.is_significant]
                if significant:
                    st.markdown("### ⚠️ Значимые потоки")
                    for f in significant[:10]:
                        direction = "📈 Приток" if f.net_flow_usd > 0 else "📉 Отток"
                        st.write(f"{direction}: **{f.token_symbol}** ({f.network}) — ${abs(f.net_flow_usd):,.0f}")
            else:
                st.info("Нет данных о потоках")


# =============================================================================
# Новые пулы
# =============================================================================

elif page == "🆕 Новые пулы":
    st.title("🆕 Новые пулы")
    
    st.info("""
    **Что это:**
    Мониторинг недавно созданных пулов с оценкой риска.
    
    **Уровни риска:**
    - 🟢 Низкий — достаточно TVL, есть позиции, токены известны
    - 🟡 Средний — умеренный риск
    - 🟠 Высокий — мало TVL или позиций
    - 🔴 Очень высокий — новый токен, мало данных
    
    **Рекомендуемый диапазон** — на основе анализа существующих позиций
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        max_age = st.slider("Макс. возраст пула (дни)", 1, 60, 30)
    with col2:
        min_tvl = st.number_input("Мин. TVL ($)", value=50000, step=10000, key="new_pools_tvl")
    
    if st.button("🔍 Найти новые пулы", type="primary"):
        pools = get_new_pools(max_age_days=max_age)
        pools = [p for p in pools if p.tvl_usd >= min_tvl]
        
        if pools:
            st.success(f"Найдено {len(pools)} новых пулов")
            
            for pool in pools[:20]:
                risk_icons = {"low": "🟢", "medium": "🟡", "high": "🟠", "very_high": "🔴"}
                risk_names = {"low": "Низкий", "medium": "Средний", "high": "Высокий", "very_high": "Очень высокий"}
                icon = risk_icons.get(pool.risk_level, "⚪")
                
                with st.expander(f"{icon} {pool.token0_symbol}/{pool.token1_symbol} ({pool.network})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("TVL", f"${pool.tvl_usd:,.0f}")
                    col2.metric("Возраст", f"{pool.age_days:.1f} дней")
                    col3.metric("Риск", risk_names.get(pool.risk_level, pool.risk_level))
                    
                    st.write(f"**Комиссия пула:** {pool.fee_tier / 10000:.2f}%")
                    st.write(f"**Позиций:** {pool.position_count}")
                    st.write(f"**Рекомендуемый диапазон:** ±{pool.recommended_range_percent:.1f}%")
                    
                    if pool.avg_holding_days > 0:
                        st.write(f"**Ср. время удержания:** {pool.avg_holding_days:.1f} дней")
        else:
            st.info("Новых пулов не найдено")


# =============================================================================
# Топ владельцев LP
# =============================================================================

elif page == "🏆 Топ владельцев LP":
    st.title("🏆 Топ владельцев LP-позиций")
    
    st.info("""
    **Что это:**
    Рейтинг LP-провайдеров по **проценту успешно закрытых позиций**.
    
    **Главная метрика — Win Rate:**
    - Считается ТОЛЬКО по **закрытым** позициям
    - Win Rate = (Прибыльные закрытые позиции) / (Все закрытые позиции) × 100%
    - Прибыльная = позиция где (выведено + комиссии) > (депозит + газ)
    
    **Важно:** Владельцы с малым числом закрытых позиций фильтруются для точности.
    """)
    
    stats = get_pool_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Позиции в базе", stats['total_positions'])
    col2.metric("Владельцев", stats['total_owners'])
    col3.metric("Пулов", stats['total_pools'])
    
    st.markdown("---")
    
    if stats['total_positions'] == 0:
        st.warning("⚠️ Нет данных о позициях. Нажмите кнопку ниже или загрузите данные.")
        
        if st.button("📥 Загрузить позиции", type="primary"):
            with st.spinner("Загрузка позиций (это может занять несколько минут)..."):
                loader = PositionLoader()
                results = loader.load_all_positions(limit_per_network=200)
                calculate_positions_usd()
                st.success(f"Загружено: {results}")
                st.rerun()
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            limit = st.slider("Показать топ N", 10, 100, 20, key="top_owners_limit")
        with col2:
            order_by = st.selectbox(
                "Сортировка",
                ["win_rate", "pnl", "positions"],
                format_func=lambda x: {
                    "win_rate": "📊 По Win Rate (% успеха)", 
                    "pnl": "💰 По прибыли", 
                    "positions": "📈 По кол-ву позиций"
                }[x],
                help="Win Rate — главная метрика успешности",
                key="top_owners_order"
            )
        with col3:
            min_closed = st.number_input(
                "Мин. закрытых позиций",
                min_value=1,
                value=3,
                step=1,
                help="Фильтр для статистической значимости Win Rate",
                key="top_owners_min_closed"
            )
        
        # Кнопка для загрузки/обновления данных
        if st.button("📊 Показать рейтинг", type="primary"):
            # Пересчитаем USD если нужно
            calculate_positions_usd()
            
            analyzer = OwnerAnalyzer()
            # Берём ВСЕХ владельцев (без лимита), потом фильтруем и ограничиваем
            all_owners = analyzer.get_top_owners(
                limit=10000,  # Большое число чтобы получить всех
                order_by=order_by,
                min_positions=1,  # Базовый фильтр
            )
            
            # Сначала фильтруем по минимальному числу закрытых позиций
            # Потом берём только нужное количество (limit)
            owners = [o for o in all_owners if o.closed_positions >= min_closed][:limit]
            
            # Сохраняем в session state для персистентности
            st.session_state['top_owners'] = owners
            st.session_state['top_owners_loaded'] = True
        
        # Отображаем owners из session_state (если есть)
        if st.session_state.get('top_owners_loaded') and st.session_state.get('top_owners'):
            owners = st.session_state['top_owners']
            
            st.success(f"Найдено {len(owners)} владельцев с {min_closed}+ закрытых позиций")
            
            # Пояснение
            st.caption("""
            🏆 Win Rate ≥ 70% | 🟢 ≥ 50% | 🟡 ≥ 30% | 🔴 < 30%
            
            **Нажмите на владельца** чтобы увидеть все его позиции и добавить в мониторинг.
            """)
            
            # Показываем каждого владельца с expander
            for i, o in enumerate(owners, 1):
                # Иконка по win rate
                if o.win_rate >= 0.7:
                    wr_icon = "🏆"
                elif o.win_rate >= 0.5:
                    wr_icon = "🟢"
                elif o.win_rate >= 0.3:
                    wr_icon = "🟡"
                else:
                    wr_icon = "🔴"
                
                header = f"{wr_icon} #{i} {o.address[:10]}...{o.address[-6:]} — Win Rate: {o.win_rate*100:.0f}% | PnL: ${o.realized_pnl_usd:,.2f}"
                
                with st.expander(header):
                    # Метрики владельца
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Win Rate", f"{o.win_rate*100:.1f}%")
                    col2.metric("Закрытых", o.closed_positions)
                    col3.metric("Прибыльных", o.profitable_positions)
                    col4.metric("PnL", f"${o.realized_pnl_usd:,.2f}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Всего позиций", o.total_positions)
                    col2.metric("Открытых", o.open_positions)
                    col3.metric("Ср. время (дни)", f"{o.avg_holding_days:.1f}")
                    col4.metric("Ср. PnL", f"${o.avg_pnl_per_position:.2f}")
                    
                    st.markdown("---")
                    
                    # Кнопки действий
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        if st.button(f"👁️ Добавить в мониторинг", key=f"watch_{o.address}"):
                            result = add_to_watchlist(o.address)
                            if result == "success":
                                st.success("✅ Добавлен в мониторинг!")
                            elif result == "exists":
                                st.warning("⚠️ Уже в списке мониторинга")
                            elif result == "not_found":
                                st.error("❌ Владелец не найден в базе данных")
                            else:
                                st.error(f"❌ Ошибка при добавлении")
                    
                    with col2:
                        st.code(o.address, language=None)
                    
                    with col3:
                        if o.favorite_networks:
                            st.write(f"**Сети:** {', '.join(o.favorite_networks[:3])}")
                    
                    # Детали позиций
                    st.markdown("#### 📋 Все позиции владельца:")
                    
                    pos_df = get_owner_positions(o.address)
                    
                    if not pos_df.empty:
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Нет данных о позициях")
            
            # Success patterns
            st.markdown("---")
            st.markdown("### 📈 Паттерны успешных LP (Win Rate ≥ 50%)")
            
            # Фильтруем только успешных
            successful_owners = [o for o in owners if o.win_rate >= 0.5]
            
            if successful_owners:
                # Статистика по успешным
                avg_wr = sum(o.win_rate for o in successful_owners) / len(successful_owners)
                avg_hold = sum(o.avg_holding_days for o in successful_owners) / len(successful_owners)
                avg_closed = sum(o.closed_positions for o in successful_owners) / len(successful_owners)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Успешных LP", len(successful_owners))
                col2.metric("Ср. Win Rate", f"{avg_wr*100:.1f}%")
                col3.metric("Ср. закрытых позиций", f"{avg_closed:.1f}")
                col4.metric("Ср. время удержания", f"{avg_hold:.1f} дней")
            else:
                st.info("Нет владельцев с Win Rate ≥ 50%")
        elif not st.session_state.get('top_owners_loaded'):
            st.info("👆 Нажмите 'Показать рейтинг' чтобы загрузить данные о владельцах LP")


# =============================================================================
# Мониторинг
# =============================================================================

elif page == "👁️ Мониторинг":
    st.title("👁️ Мониторинг успешных LP")
    
    st.info("""
    **Что это:**
    Отслеживание выбранных LP-провайдеров в реальном времени.
    
    **Возможности:**
    - 📊 Следить за позициями успешных владельцев
    - 🔔 Получать уведомления о новых/закрытых позициях
    - 📱 Telegram-алерты при активности
    - 🤖 Автоматический мониторинг каждые 10 минут (через scheduler.py)
    
    **Как добавить:** Перейдите в "Топ владельцев LP" и нажмите "Добавить в мониторинг"
    """)
    
    # Статус автоматического мониторинга
    st.markdown("### 🤖 Автоматический мониторинг")
    
    settings = get_settings()
    telegram_status = "✅ Настроен" if settings.telegram.enabled else "❌ Не настроен"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Статус Telegram:** {telegram_status}
        
        **Как запустить автомониторинг:**
        ```bash
        python scripts/scheduler.py
        ```
        
        Scheduler проверяет всех отслеживаемых владельцев **каждые 10 минут** 
        и отправляет Telegram-уведомления при изменениях.
        """)
    
    with col2:
        # Показываем статистику последней проверки
        with session_scope() as session:
            last_checks = session.query(WatchedOwner.last_checked_at).filter(
                WatchedOwner.last_checked_at.isnot(None)
            ).order_by(WatchedOwner.last_checked_at.desc()).first()
            
            if last_checks and last_checks[0]:
                last_check_time = last_checks[0]
                st.metric("Последняя проверка", last_check_time.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("Последняя проверка", "Никогда")
    
    st.markdown("---")
    
    watched = get_watched_owners()
    
    st.markdown(f"### 📋 Отслеживаемые владельцы ({len(watched)})")
    
    if not watched:
        st.warning("Список пуст. Добавьте владельцев из раздела «Топ владельцев LP».")
    else:
        # Показываем всех отслеживаемых
        for w in watched:
            # Определяем иконку активности
            if w["new_activity"]:
                activity_icon = "🔴"
                activity_text = "НОВАЯ АКТИВНОСТЬ!"
            else:
                activity_icon = "🟢"
                activity_text = "Без изменений"
            
            header = f"{activity_icon} {w['address'][:10]}...{w['address'][-6:]} — {activity_text}"
            
            with st.expander(header, expanded=w["new_activity"]):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Всего позиций", w["current_count"])
                col2.metric("Открытых", w["open_positions"])
                col3.metric("Закрытых", w["closed_positions"])
                col4.metric("Добавлен", w["added_at"].strftime("%Y-%m-%d") if w["added_at"] else "-")
                
                if w["note"]:
                    st.write(f"**Заметка:** {w['note']}")
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Обновить позиции", key=f"refresh_{w['address']}"):
                        st.info("Для обновления перейдите в 'Загрузка данных' и загрузите позиции")
                
                with col2:
                    telegram_status = "✅ Вкл" if w["notify_telegram"] else "❌ Выкл"
                    st.write(f"**Telegram:** {telegram_status}")
                
                with col3:
                    if st.button("🗑️ Удалить", key=f"delete_{w['address']}"):
                        if remove_from_watchlist(w["address"]):
                            st.success("Удалён из мониторинга")
                            st.rerun()
                
                # Показываем позиции
                st.markdown("#### 📋 Позиции владельца:")
                st.code(w["address"], language=None)
                pos_df = get_owner_positions(w["address"])
                
                if not pos_df.empty:
                    # Выделяем открытые позиции
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Нет данных о позициях")
    
    st.markdown("---")
    
    # Ручное добавление
    st.markdown("### ➕ Добавить вручную")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_address = st.text_input(
            "Адрес кошелька",
            placeholder="0x...",
            help="Введите полный адрес кошелька для отслеживания"
        )
    with col2:
        manual_note = st.text_input("Заметка", placeholder="Описание")
    
    if st.button("➕ Добавить в мониторинг"):
        if manual_address and len(manual_address) == 42:
            result = add_to_watchlist(manual_address, manual_note)
            if result == "success":
                st.success(f"✅ Добавлен: {manual_address[:10]}...")
                st.rerun()
            elif result == "exists":
                st.warning("⚠️ Уже в списке мониторинга")
            elif result == "not_found":
                st.error("❌ Владелец не найден в базе. Сначала загрузите позиции.")
            else:
                st.error("❌ Ошибка при добавлении")
        else:
            st.error("Введите корректный адрес (42 символа)")
    
    st.markdown("---")
    
    # Проверка активности
    st.markdown("### 🔔 Проверка активности")
    
    st.write("""
    Нажмите кнопку ниже чтобы проверить новую активность у всех отслеживаемых владельцев.
    При обнаружении новых позиций будет отправлено уведомление в Telegram (если настроен).
    """)
    
    if st.button("🔍 Проверить активность", type="primary"):
        with st.spinner("Проверка активности..."):
            new_activity_found = False
            
            for w in watched:
                if w["new_activity"]:
                    new_activity_found = True
                    diff = w["current_count"] - w["last_count"]
                    
                    if diff > 0:
                        msg = f"🆕 {w['address'][:10]}... открыл {diff} новых позиций!"
                    else:
                        msg = f"🔒 {w['address'][:10]}... закрыл {abs(diff)} позиций!"
                    
                    st.warning(msg)
                    
                    # Отправляем в Telegram
                    settings = get_settings()
                    if settings.telegram.enabled and w["notify_telegram"]:
                        try:
                            from src.signals.telegram import send_telegram_message
                            send_telegram_message(f"LP Мониторинг: {msg}")
                        except:
                            pass
            
            if not new_activity_found:
                st.success("✅ Новой активности не обнаружено")
            
            # Обновляем last_position_count
            with session_scope() as session:
                for w in watched:
                    watched_obj = session.query(WatchedOwner).filter(
                        WatchedOwner.owner_address == w["address"]
                    ).first()
                    if watched_obj:
                        watched_obj.last_position_count = w["current_count"]
                        watched_obj.last_checked_at = datetime.now()
                session.commit()


# =============================================================================
# Сигналы
# =============================================================================

elif page == "⚠️ Сигналы":
    st.title("⚠️ Сигналы и уведомления")
    
    st.info("""
    **Что это:**
    История всех обнаруженных событий (оттоки, потоки, новые пулы).
    
    **Telegram:**
    Настройте бота для получения уведомлений на телефон.
    """)
    
    df = get_signals(limit=50)
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего сигналов", len(df))
        critical_count = len(df[df["Важность"] == "Критический"])
        col2.metric("Критических", critical_count)
        sent_count = len(df[df["Отправлен"] == "✅"])
        col3.metric("Отправлено", sent_count)
        
        st.markdown("---")
        
        severity_filter = st.multiselect(
            "Фильтр по важности",
            ["Критический", "Внимание", "Инфо"],
            default=["Критический", "Внимание", "Инфо"],
        )
        
        filtered = df[df["Важность"].isin(severity_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.info("Сигналов пока нет. Запустите анализ на других страницах.")
    
    st.markdown("---")
    
    # Telegram setup
    st.markdown("### 📱 Telegram уведомления")
    
    settings = get_settings()
    
    if settings.telegram.enabled:
        st.success("✅ Telegram настроен")
        
        if st.button("Отправить тестовое сообщение"):
            from src.signals.telegram import send_telegram_message
            if send_telegram_message("🔔 Тестовое сообщение от Revert LP Strategy!"):
                st.success("Сообщение отправлено!")
            else:
                st.error("Ошибка отправки")
    else:
        st.warning("⚠️ Telegram не настроен")
        
        with st.expander("Как настроить Telegram?"):
            st.markdown("""
            **Шаг 1:** Создайте бота
            - Откройте [@BotFather](https://t.me/botfather) в Telegram
            - Отправьте `/newbot` и следуйте инструкциям
            - Скопируйте токен бота
            
            **Шаг 2:** Получите Chat ID
            - Напишите что-нибудь вашему боту
            - Откройте: `https://api.telegram.org/bot<TOKEN>/getUpdates`
            - Найдите `chat.id` в ответе
            
            **Шаг 3:** Добавьте в `.env`:
            ```
            TELEGRAM_BOT_TOKEN=ваш_токен
            TELEGRAM_CHAT_ID=ваш_chat_id
            ```
            
            **Шаг 4:** Перезапустите приложение
            """)


# =============================================================================
# Настройки
# =============================================================================

elif page == "⚙️ Настройки":
    st.title("⚙️ Настройки")
    
    settings = get_settings()
    
    st.markdown("### 🔧 Текущие настройки")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Фильтры пулов:**")
        st.write(f"• Мин. TVL: ${settings.pool_filter.min_tvl_usd:,.0f}")
        st.write(f"• Мин. объём 24ч: ${settings.pool_filter.min_volume_24h_usd:,.0f}")
        
        st.markdown("**Детекция оттоков:**")
        st.write(f"• Порог оттока: ${settings.capital_flow.large_outflow_usd:,.0f}")
        st.write(f"• Порог % TVL: {settings.capital_flow.large_outflow_tvl_percent}%")
    
    with col2:
        st.markdown("**Анализ владельцев:**")
        st.write(f"• Мин. позиций: {settings.owner_analysis.min_positions}")
        st.write(f"• Топ для анализа: {settings.owner_analysis.top_owners_count}")
        
        st.markdown("**Новые пулы:**")
        st.write(f"• Макс. возраст: {settings.new_token.max_age_days} дней")
    
    st.markdown("---")
    
    st.markdown("### 🗄️ База данных")
    st.write(f"Путь: `{settings.database.url}`")
    
    stats = get_pool_stats()
    st.write(f"• Пулов: {stats['total_pools']}")
    st.write(f"• Токенов: {stats['total_tokens']}")
    st.write(f"• Свопов: {stats['total_swaps']}")
    st.write(f"• Позиций: {stats['total_positions']}")
    st.write(f"• Владельцев: {stats['total_owners']}")
    
    st.markdown("---")
    
    st.info("""
    **Как изменить настройки:**
    
    Отредактируйте файл `config/settings.py` или используйте переменные окружения в `.env`
    """)
    
    with st.expander("Очистить базу данных"):
        st.warning("⚠️ Это удалит ВСЕ загруженные данные!")
        if st.button("🗑️ Очистить базу", type="secondary"):
            from src.db.database import reset_db
            reset_db()
            st.success("База данных очищена. Перезагрузите страницу.")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
