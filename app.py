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
import logging
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)

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


def get_network_stats_table() -> pd.DataFrame:
    """Get statistics by network for all entity types."""
    with session_scope() as session:
        from sqlalchemy import func, distinct
        
        # Get all networks that have any data
        all_networks = set()
        
        # Pools by network
        pools_by_network = dict(
            session.query(Pool.network, func.count(Pool.id))
            .group_by(Pool.network)
            .all()
        )
        all_networks.update(pools_by_network.keys())
        
        # Swaps by network
        swaps_by_network = dict(
            session.query(Swap.network, func.count(Swap.id))
            .group_by(Swap.network)
            .all()
        )
        all_networks.update(swaps_by_network.keys())
        
        # Positions by network
        positions_by_network = dict(
            session.query(Position.network, func.count(Position.id))
            .group_by(Position.network)
            .all()
        )
        all_networks.update(positions_by_network.keys())
        
        # Owners by network (count distinct owner addresses per network from positions)
        owners_by_network = dict(
            session.query(Position.network, func.count(distinct(Position.owner_address)))
            .filter(Position.owner_address != None)
            .group_by(Position.network)
            .all()
        )
        all_networks.update(owners_by_network.keys())
        
        # Build table data
        data = []
        for network in sorted(all_networks):
            data.append({
                "Сеть": network,
                "Пулы": pools_by_network.get(network, 0),
                "Свопы": swaps_by_network.get(network, 0),
                "Позиции": positions_by_network.get(network, 0),
                "Владельцы": owners_by_network.get(network, 0),
            })
        
        # Add totals row
        if data:
            data.append({
                "Сеть": "ИТОГО",
                "Пулы": sum(d["Пулы"] for d in data),
                "Свопы": sum(d["Свопы"] for d in data),
                "Позиции": sum(d["Позиции"] for d in data),
                "Владельцы": session.query(Owner).count(),  # Total unique owners
            })
        
        return pd.DataFrame(data)


def save_period_stats_to_db(stats: dict, networks: list, min_tvl: float) -> None:
    """Save period statistics to database."""
    from src.db.models import PeriodStatistics
    
    with session_scope() as session:
        # Delete old statistics for these networks and min_tvl
        session.query(PeriodStatistics).filter(
            PeriodStatistics.network.in_(networks),
            PeriodStatistics.min_tvl == min_tvl
        ).delete()
        
        # Save new statistics
        for stat_type in ["positions", "swaps", "owners"]:
            df = stats.get(stat_type, pd.DataFrame())
            if df.empty:
                continue
            
            for _, row in df.iterrows():
                network = row["Сеть"]
                if network == "ИТОГО":
                    continue
                
                for period_name in df.columns:
                    if period_name == "Сеть":
                        continue
                    
                    count = int(row.get(period_name, 0))
                    stat = PeriodStatistics(
                        network=network,
                        period_name=period_name,
                        stat_type=stat_type,
                        count=count,
                        min_tvl=min_tvl
                    )
                    session.add(stat)
        
        session.commit()
        logger.info(f"Saved period statistics for networks {networks} with min_tvl={min_tvl}")


def load_period_stats_from_db(networks: list, min_tvl: float) -> dict:
    """Load period statistics from database."""
    from src.db.models import PeriodStatistics
    
    periods = [
        "Последняя неделя",
        "Последний месяц",
        "Последние 3 месяца",
        "Последние 4 месяца",
        "Последние 6 месяцев",
        "Последний год",
        "Последние 2 года",
    ]
    
    with session_scope() as session:
        stats = session.query(PeriodStatistics).filter(
            PeriodStatistics.network.in_(networks),
            PeriodStatistics.min_tvl == min_tvl
        ).all()
        
        if not stats:
            return None
        
        # Build data structures
        positions_data = {net: {p: 0 for p in periods} for net in networks}
        swaps_data = {net: {p: 0 for p in periods} for net in networks}
        owners_data = {net: {p: 0 for p in periods} for net in networks}
        
        for stat in stats:
            if stat.stat_type == "positions":
                positions_data[stat.network][stat.period_name] = stat.count
            elif stat.stat_type == "swaps":
                swaps_data[stat.network][stat.period_name] = stat.count
            elif stat.stat_type == "owners":
                owners_data[stat.network][stat.period_name] = stat.count
        
        # Build DataFrames
        positions_rows = []
        swaps_rows = []
        owners_rows = []
        
        for network in networks:
            positions_rows.append({"Сеть": network, **positions_data[network]})
            swaps_rows.append({"Сеть": network, **swaps_data[network]})
            owners_rows.append({"Сеть": network, **owners_data[network]})
        
        # Add totals
        totals_pos = {"Сеть": "ИТОГО"}
        totals_swaps = {"Сеть": "ИТОГО"}
        totals_owners = {"Сеть": "ИТОГО"}
        
        for period_name in periods:
            totals_pos[period_name] = sum(r.get(period_name, 0) for r in positions_rows)
            totals_swaps[period_name] = sum(r.get(period_name, 0) for r in swaps_rows)
            totals_owners[period_name] = sum(r.get(period_name, 0) for r in owners_rows)
        
        positions_rows.append(totals_pos)
        swaps_rows.append(totals_swaps)
        owners_rows.append(totals_owners)
        
        return {
            "positions": pd.DataFrame(positions_rows),
            "swaps": pd.DataFrame(swaps_rows),
            "owners": pd.DataFrame(owners_rows),
        }


def fetch_period_stats_from_graph(networks: list, min_tvl: float = 50000, progress_callback=None) -> dict:
    """Fetch period statistics by querying each pool individually from The Graph API."""
    from datetime import datetime, timedelta
    from src.data.subgraph import SubgraphClient
    from src.db.models import Pool
    
    periods = [
        ("Последняя неделя", 7),
        ("Последний месяц", 30),
        ("Последние 3 месяца", 90),
        ("Последние 4 месяца", 120),
        ("Последние 6 месяцев", 180),
        ("Последний год", 365),
        ("Последние 2 года", 730),
    ]
    
    now = datetime.utcnow()
    
    # Get pools from database filtered by network and TVL
    # Extract all needed data inside session context to avoid DetachedInstanceError
    pool_data = []
    with session_scope() as session:
        pools = session.query(Pool).filter(
            Pool.network.in_(networks),
            Pool.tvl_usd >= min_tvl
        ).order_by(Pool.tvl_usd.desc()).all()
        
        # Extract all needed attributes while session is active
        for pool in pools:
            pool_data.append({
                "network": pool.network,
                "address": pool.address.lower(),
                "token0_symbol": pool.token0_symbol or "",
                "token1_symbol": pool.token1_symbol or "",
            })
    
    if not pool_data:
        logger.warning(f"No pools found for networks {networks} with TVL >= {min_tvl}")
        # Return empty dataframes
        periods_names = [p[0] for p in periods]
        empty_df = pd.DataFrame([{"Сеть": "Нет данных"}] + [{col: 0 for col in periods_names}])
        return {
            "positions": empty_df,
            "swaps": empty_df,
            "owners": empty_df,
        }
    
    # Initialize data structures per network
    network_positions = {net: {p[0]: 0 for p in periods} for net in networks}
    network_swaps = {net: {p[0]: 0 for p in periods} for net in networks}
    network_owners = {net: {p[0]: set() for p in periods} for net in networks}
    
    total_pools = len(pool_data)
    current_pool = 0
    
    # Group pools by network
    pools_by_network = {}
    for pool_info in pool_data:
        network = pool_info["network"]
        if network not in pools_by_network:
            pools_by_network[network] = []
        pools_by_network[network].append(pool_info)
    
    # Query each pool for each period
    for network, network_pools in pools_by_network.items():
        try:
            client = SubgraphClient(network)
        except ValueError as e:
            logger.error(f"Failed to create client for {network}: {e}")
            continue
        
        network_pool_count = len(network_pools)
        network_pool_index = 0
        
        for pool_info in network_pools:
            current_pool += 1
            network_pool_index += 1
            pool_address = pool_info["address"]
            
            if progress_callback:
                progress_callback(
                    current_pool / total_pools,
                    f"{network}: {pool_info['token0_symbol']}/{pool_info['token1_symbol']} ({current_pool}/{total_pools} пулов)"
                )
            
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                start_time = int(cutoff.timestamp())
                
                # Query mints (positions) for this pool
                try:
                    mints_query = """
                    query getPoolMints($poolId: String!, $startTime: BigInt!, $first: Int!) {
                        mints(
                            first: $first
                            where: { pool: $poolId, timestamp_gte: $startTime }
                            orderBy: timestamp
                            orderDirection: desc
                        ) {
                            id
                            owner
                        }
                    }
                    """
                    result = client.query(mints_query, {
                        "poolId": pool_address,
                        "startTime": str(start_time),
                        "first": 1000
                    })
                    mints = result.get("mints", [])
                    network_positions[network][period_name] += len(mints)
                    for m in mints:
                        if m.get("owner"):
                            network_owners[network][period_name].add(m["owner"].lower())
                except Exception as e:
                    logger.debug(f"Error fetching mints for pool {pool_address}/{period_name}: {e}")
                
                # Query swaps for this pool
                try:
                    swaps_query = """
                    query getPoolSwaps($poolId: String!, $startTime: BigInt!, $first: Int!) {
                        swaps(
                            first: $first
                            where: { pool: $poolId, timestamp_gte: $startTime }
                            orderBy: timestamp
                            orderDirection: desc
                        ) {
                            id
                        }
                    }
                    """
                    result = client.query(swaps_query, {
                        "poolId": pool_address,
                        "startTime": str(start_time),
                        "first": 1000
                    })
                    swaps = result.get("swaps", [])
                    network_swaps[network][period_name] += len(swaps)
                except Exception as e:
                    logger.debug(f"Error fetching swaps for pool {pool_address}/{period_name}: {e}")
    
    # Build result dataframes
    positions_data = []
    swaps_data = []
    owners_data = []
    
    for network in networks:
        positions_row = {"Сеть": network}
        swaps_row = {"Сеть": network}
        owners_row = {"Сеть": network}
        
        for period_name, _ in periods:
            positions_row[period_name] = network_positions[network].get(period_name, 0)
            swaps_row[period_name] = network_swaps[network].get(period_name, 0)
            owners_row[period_name] = len(network_owners[network].get(period_name, set()))
        
        positions_data.append(positions_row)
        swaps_data.append(swaps_row)
        owners_data.append(owners_row)
    
    # Add totals
    if positions_data:
        totals_pos = {"Сеть": "ИТОГО"}
        totals_swaps = {"Сеть": "ИТОГО"}
        totals_owners = {"Сеть": "ИТОГО"}
        
        # For owners, we need to merge sets across networks to avoid double counting
        all_owners_by_period = {p[0]: set() for p in periods}
        for network in networks:
            for period_name, _ in periods:
                all_owners_by_period[period_name].update(network_owners[network].get(period_name, set()))
        
        for period_name, _ in periods:
            totals_pos[period_name] = sum(r.get(period_name, 0) for r in positions_data)
            totals_swaps[period_name] = sum(r.get(period_name, 0) for r in swaps_data)
            totals_owners[period_name] = len(all_owners_by_period[period_name])
        
        positions_data.append(totals_pos)
        swaps_data.append(totals_swaps)
        owners_data.append(totals_owners)
    
    result = {
        "positions": pd.DataFrame(positions_data),
        "swaps": pd.DataFrame(swaps_data),
        "owners": pd.DataFrame(owners_data),
    }
    
    # Save to database
    try:
        save_period_stats_to_db(result, networks, min_tvl)
    except Exception as e:
        logger.error(f"Error saving period statistics to DB: {e}", exc_info=True)
    
    return result


def get_period_stats_table() -> dict:
    """Get statistics by time period for Positions and Swaps."""
    from datetime import datetime, timedelta
    
    periods = [
        ("Последняя неделя", 7),
        ("Последний месяц", 30),
        ("Последние 3 месяца", 90),
        ("Последние 4 месяца", 120),
        ("Последние 6 месяцев", 180),
        ("Последний год", 365),
        ("Последние 2 года", 730),
    ]
    
    with session_scope() as session:
        from sqlalchemy import func, distinct
        
        now = datetime.utcnow()
        
        # Get all networks
        all_networks = set()
        for net, in session.query(Position.network).distinct().all():
            all_networks.add(net)
        for net, in session.query(Swap.network).distinct().all():
            all_networks.add(net)
        
        # Build positions data
        positions_data = []
        for network in sorted(all_networks):
            row = {"Сеть": network}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                count = session.query(func.count(Position.id)).filter(
                    Position.network == network,
                    Position.created_at >= cutoff
                ).scalar() or 0
                row[period_name] = count
            positions_data.append(row)
        
        # Add totals for positions
        if positions_data:
            totals_row = {"Сеть": "ИТОГО"}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                total = session.query(func.count(Position.id)).filter(
                    Position.created_at >= cutoff
                ).scalar() or 0
                totals_row[period_name] = total
            positions_data.append(totals_row)
        
        # Build swaps data
        swaps_data = []
        for network in sorted(all_networks):
            row = {"Сеть": network}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                count = session.query(func.count(Swap.id)).filter(
                    Swap.network == network,
                    Swap.timestamp >= cutoff
                ).scalar() or 0
                row[period_name] = count
            swaps_data.append(row)
        
        # Add totals for swaps
        if swaps_data:
            totals_row = {"Сеть": "ИТОГО"}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                total = session.query(func.count(Swap.id)).filter(
                    Swap.timestamp >= cutoff
                ).scalar() or 0
                totals_row[period_name] = total
            swaps_data.append(totals_row)
        
        # Build owners data (unique owners who created positions in period)
        owners_data = []
        for network in sorted(all_networks):
            row = {"Сеть": network}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                count = session.query(func.count(distinct(Position.owner_address))).filter(
                    Position.network == network,
                    Position.created_at >= cutoff,
                    Position.owner_address != None
                ).scalar() or 0
                row[period_name] = count
            owners_data.append(row)
        
        # Add totals for owners
        if owners_data:
            totals_row = {"Сеть": "ИТОГО"}
            for period_name, days in periods:
                cutoff = now - timedelta(days=days)
                total = session.query(func.count(distinct(Position.owner_address))).filter(
                    Position.created_at >= cutoff,
                    Position.owner_address != None
                ).scalar() or 0
                totals_row[period_name] = total
            owners_data.append(totals_row)
        
        return {
            "positions": pd.DataFrame(positions_data),
            "swaps": pd.DataFrame(swaps_data),
            "owners": pd.DataFrame(owners_data),
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


def get_period_hours(period_name: str) -> int:
    """Convert period name to hours."""
    periods = {
        "Последняя неделя": 7 * 24,  # 168 hours
        "Последний месяц": 30 * 24,  # 720 hours
        "Последние 3 месяца": 90 * 24,  # 2160 hours
        "Последние 4 месяца": 120 * 24,  # 2880 hours
        "Последний год": 365 * 24,  # 8760 hours
        "Последние 2 года": 730 * 24,  # 17520 hours
        "Последние 3 года": 1095 * 24,  # 26280 hours
        "Последние 4 года": 1460 * 24,  # 35040 hours
        "Последние 5 лет": 1825 * 24,  # 43800 hours
    }
    return periods.get(period_name, 168)  # Default to 1 week


def load_all_data_action(networks: list, min_tvl: float, positions_limit: int, hours: int = 168):
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
    
    total_loaded = 0
    detailed_errors = []
    
    # Check GRAPH_API_KEY first
    if not GRAPH_API_KEY:
        status.error("❌ GRAPH_API_KEY не установлен!")
        status.text("💡 Добавьте GRAPH_API_KEY в файл .env")
        results["pools_error"] = "GRAPH_API_KEY не установлен"
        return results
    
    for i, network in enumerate(networks):
        try:
            status.text(f"📊 Шаг 1/4: Загрузка пулов из {network}...")
            
            # Check if network is enabled
            network_config = NETWORKS.get(network)
            if not network_config:
                error_msg = f"Сеть '{network}' не найдена в конфигурации"
                results["pools"][network] = f"Ошибка: {error_msg}"
                detailed_errors.append(f"{network}: {error_msg}")
                logger.error(error_msg)
                continue
            
            if not network_config.enabled:
                error_msg = f"Сеть '{network}' отключена"
                results["pools"][network] = f"Ошибка: {error_msg}"
                detailed_errors.append(f"{network}: {error_msg}")
                logger.error(error_msg)
                continue
            
            # Check if subgraph is configured
            if not network_config.subgraphs.uniswap_v3:
                error_msg = f"Subgraph для {network} не настроен"
                results["pools"][network] = f"Ошибка: {error_msg}"
                detailed_errors.append(f"{network}: {error_msg}")
                logger.error(error_msg)
                continue
            
            with session_scope() as session:
                count = loader.load_pools_for_network(session, network, min_tvl=min_tvl)
                results["pools"][network] = count
                total_loaded += count
                
                if count > 0:
                    status.text(f"📊 Шаг 1/4: Загружено {total_loaded} пулов... ({network}: {count})")
                else:
                    status.text(f"⚠️ Шаг 1/4: {network}: пулы не найдены (TVL >= ${min_tvl:,.0f})")
                
                # Ensure commit happens
                session.commit()
        except ValueError as e:
            # This is usually GRAPH_API_KEY or subgraph configuration error
            error_msg = str(e)
            results["pools"][network] = f"Ошибка конфигурации: {error_msg[:50]}"
            detailed_errors.append(f"{network}: {error_msg}")
            logger.error(f"Configuration error for {network}: {e}", exc_info=True)
        except Exception as e:
            error_msg = str(e)[:100]
            results["pools"][network] = f"Ошибка: {error_msg}"
            detailed_errors.append(f"{network}: {error_msg}")
            logger.error(f"Error loading pools from {network}: {e}", exc_info=True)
        progress.progress((i + 1) / len(networks) * 0.25)
    
    # Store detailed errors
    if detailed_errors:
        results["pools_detailed_errors"] = detailed_errors
    
    # Verify pools were actually saved
    with session_scope() as session:
        verify_count = session.query(Pool).filter(
            Pool.network.in_(networks),
            Pool.tvl_usd >= min_tvl
        ).count()
        if verify_count == 0 and total_loaded > 0:
            status.warning(f"⚠️ Загружено {total_loaded} пулов, но они не найдены в базе. Возможно проблема с коммитом.")
        elif verify_count > 0:
            status.text(f"✅ Подтверждено: {verify_count} пулов в базе для выбранных сетей")
    
    if total_loaded == 0:
        status.error("❌ Не удалось загрузить пулы!")
        
        # Show specific error messages
        if detailed_errors:
            status.text("Детали ошибок:")
            for err in detailed_errors[:3]:
                status.text(f"  • {err}")
        
        # Provide helpful suggestions
        suggestions = []
        if not GRAPH_API_KEY:
            suggestions.append("1. Установите GRAPH_API_KEY в файле .env")
        if detailed_errors:
            if any("не найдена" in e for e in detailed_errors):
                suggestions.append("2. Проверьте названия сетей (ethereum, arbitrum, polygon, etc.)")
            if any("Subgraph" in e for e in detailed_errors):
                suggestions.append("3. Проверьте конфигурацию subgraph в config/networks.py")
            if any("TVL" in e for e in detailed_errors):
                suggestions.append("4. Попробуйте уменьшить 'Мин. TVL пула'")
        
        if suggestions:
            status.text("💡 Рекомендации:")
            for suggestion in suggestions:
                status.text(f"  {suggestion}")
        
        results["pools_error"] = True
    
    # Шаг 2: Загрузка свопов
    status.text(f"💱 Шаг 2/4: Загрузка свопов (период: {hours // 24} дней)...")
    
    # Small delay to ensure Step 1 commits are visible
    import time
    time.sleep(0.5)
    
    with session_scope() as session:
        # First check if any pools exist at all
        total_pools = session.query(Pool).count()
        pools_in_networks = session.query(Pool).filter(
            Pool.network.in_(networks)
        ).count()
        pools_above_tvl = session.query(Pool).filter(
            Pool.tvl_usd >= min_tvl,
            Pool.network.in_(networks)
        ).count()
        
        # Debug: Show what networks are in DB
        all_networks_in_db = [n[0] for n in session.query(Pool.network).distinct().all()]
        logger.info(f"Step 2: Total pools={total_pools}, In networks={pools_in_networks}, Above TVL={pools_above_tvl}")
        logger.info(f"Step 2: Networks in DB: {all_networks_in_db}, Looking for: {networks}")
        
        pools = session.query(Pool).filter(
            Pool.tvl_usd >= min_tvl,
            Pool.network.in_(networks)
        ).order_by(Pool.tvl_usd.desc()).limit(30).all()
        
        if not pools:
            # Provide detailed diagnostic information
            if total_pools == 0:
                status.error("❌ В базе данных нет пулов!")
                status.text("💡 Решение: Убедитесь, что Шаг 1 (Загрузка пулов) выполнен успешно.")
                results["swap_warning"] = "Пулы не загружены"
                results["swap_diagnostic"] = {
                    "total_pools": 0,
                    "suggestion": "Загрузите пулы на Шаге 1"
                }
            elif pools_in_networks == 0:
                status.warning(f"⚠️ В базе нет пулов для сетей: {', '.join(networks)}")
                status.text(f"Всего пулов в базе: {total_pools}")
                if all_networks_in_db:
                    status.text(f"Доступные сети в базе: {', '.join(all_networks_in_db)}")
                    status.text(f"💡 Возможно несоответствие названий сетей. Проверьте, что выбранные сети совпадают с загруженными.")
                results["swap_warning"] = f"Нет пулов для сетей {', '.join(networks)}"
                results["swap_diagnostic"] = {
                    "total_pools": total_pools,
                    "available_networks": all_networks_in_db,
                    "requested_networks": networks,
                    "suggestion": f"Выберите сети: {', '.join(all_networks_in_db)}" if all_networks_in_db else "Загрузите пулы для выбранных сетей"
                }
            elif pools_above_tvl == 0:
                # Get TVL stats for pools in selected networks
                tvl_stats = session.query(
                    Pool.tvl_usd
                ).filter(
                    Pool.network.in_(networks)
                ).order_by(Pool.tvl_usd.desc()).all()
                
                max_tvl_val = float(tvl_stats[0][0]) if tvl_stats and tvl_stats[0][0] else 0
                min_tvl_val = float(tvl_stats[-1][0]) if tvl_stats and tvl_stats[-1][0] else 0
                
                status.warning(f"⚠️ Нет пулов с TVL >= ${min_tvl:,.0f}")
                status.text(f"Пулов в выбранных сетях: {pools_in_networks}")
                status.text(f"Максимальный TVL: ${max_tvl_val:,.0f}")
                status.text(f"Минимальный TVL: ${min_tvl_val:,.0f}")
                status.text(f"💡 Попробуйте уменьшить 'Мин. TVL пула' до ${max_tvl_val * 0.9:,.0f} или меньше")
                
                results["swap_warning"] = f"TVL фильтр слишком высокий"
                results["swap_diagnostic"] = {
                    "pools_in_networks": pools_in_networks,
                    "max_tvl": max_tvl_val,
                    "min_tvl": min_tvl_val,
                    "requested_min_tvl": min_tvl,
                    "suggestion": f"Уменьшите мин. TVL до ${max_tvl_val * 0.9:,.0f} или меньше"
                }
            else:
                results["swap_warning"] = "Пулы не найдены"
        else:
            status.text(f"💱 Шаг 2/4: Загрузка свопов из {len(pools)} пулов...")
            swap_loader = SwapLoader()
            swap_errors = []
            successful_pools = 0
            pools_with_swaps = []
            pools_without_swaps = []
            
            for i, pool in enumerate(pools):
                try:
                    count = swap_loader.load_swaps_for_pool(session, pool, hours=hours, limit=50)
                    results["swaps"] += count
                    if count > 0:
                        successful_pools += 1
                        pools_with_swaps.append(f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol} ({count})")
                    else:
                        pools_without_swaps.append(f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol}")
                except Exception as e:
                    error_msg = f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol}: {str(e)[:50]}"
                    swap_errors.append(error_msg)
                    logger.error(f"Error loading swaps for pool {pool.address}: {e}", exc_info=True)
                progress.progress(0.25 + (i + 1) / len(pools) * 0.25)
            
            results["pools_with_swaps"] = pools_with_swaps
            results["pools_without_swaps"] = pools_without_swaps
            
            if swap_errors:
                results["swap_errors"] = swap_errors
            results["successful_swap_pools"] = successful_pools
            results["total_swap_pools"] = len(pools)
    
    # Шаг 3: Загрузка позиций (открытые + закрытые через mints/burns)
    status.text("📍 Шаг 3/4: Загрузка позиций через события mint/burn...")
    pos_loader = PositionLoader()
    
    for i, network in enumerate(networks):
        try:
            with session_scope() as session:
                # Загружает открытые И закрытые позиции через анализ mint/burn событий
                result = pos_loader.load_positions_from_events(
                    session, network, min_amount_usd="100", limit=positions_limit, hours=hours
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
        # Test API key by trying to create a client
        try:
            from src.data.subgraph import SubgraphClient
            test_client = SubgraphClient("ethereum")
            st.sidebar.caption(f"🔗 Endpoint: {test_client.endpoint[:50]}...")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Проблема с API: {str(e)[:40]}")
    else:
        st.sidebar.error("❌ GRAPH_API_KEY не установлен")
        st.sidebar.caption("Добавьте в .env файл")

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
    
    st.info("""
    **Как это работает:**
    - Данные загружаются из The Graph (децентрализованный индексер блокчейнов)
    - Всё сохраняется в локальную базу данных (SQLite)
    - При следующем запуске данные уже есть — нужно только обновить
    - Дубликаты автоматически исключаются при подсчёте
    
    **Рекомендация:** Нажмите «🚀 Загрузить всё» для первичной загрузки
    """)
    
    # Professional CSS styling
    st.markdown("""
    <style>
    .stats-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a3d 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stats-header h3 {
        color: white;
        margin: 0;
    }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #3a3a5e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .section-divider {
        border-top: 2px solid #3a3a5e;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Summary metrics at top
    stats = get_pool_stats()
    
    st.markdown("### 📊 Сводка данных")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏊 Пулы", f"{stats['total_pools']:,}")
    col2.metric("💱 Свопы", f"{stats['total_swaps']:,}")
    col3.metric("📍 Позиции", f"{stats['total_positions']:,}")
    col4.metric("👥 Владельцы", f"{stats['total_owners']:,}")
    
    st.markdown("---")
    
    # Network breakdown table
    st.markdown("### 🌐 Данные по сетям")
    
    network_stats_df = get_network_stats_table()
    
    if network_stats_df.empty:
        st.warning("⚠️ База данных пуста. Загрузите данные ниже.")
    else:
        def style_network_table(row):
            if row["Сеть"] == "ИТОГО":
                return ["background-color: #1e3a5f; font-weight: bold; color: white"] * len(row)
            return [""] * len(row)
        
        styled_df = network_stats_df.style.apply(style_network_table, axis=1)
        styled_df = styled_df.format({
            "Пулы": "{:,}",
            "Свопы": "{:,}",
            "Позиции": "{:,}",
            "Владельцы": "{:,}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=200)
    
    st.markdown("---")
    
    # Period statistics section
    st.markdown("### 📅 Статистика по периодам")
    st.caption("Реальные данные из The Graph для каждого временного периода")
    
    # Network selection for period stats
    available_networks = [n for n, c in NETWORKS.items() if c.enabled]
    
    col_nets, col_tvl, col_btn = st.columns([2, 1, 1])
    with col_nets:
        period_networks = st.multiselect(
            "Сети для статистики",
            available_networks,
            default=["ethereum", "arbitrum"] if "ethereum" in available_networks else available_networks[:2],
            key="period_stats_networks"
        )
    with col_tvl:
        period_min_tvl = st.number_input(
            "Мин. TVL пула ($)",
            min_value=10000,
            value=100000,
            step=10000,
            key="period_stats_min_tvl",
            help="Используются только пулы с TVL >= этого значения"
        )
    with col_btn:
        refresh_periods = st.button("🔄 Загрузить статистику", type="primary", use_container_width=True)
    
    st.caption("⚡ Загружает актуальные данные напрямую из The Graph API по каждому пулу")
    
    # Fetch period stats from The Graph when button is clicked
    if refresh_periods:
        if not period_networks:
            st.error("Выберите хотя бы одну сеть")
        elif not GRAPH_API_KEY:
            st.error("GRAPH_API_KEY не настроен")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, text):
                progress_bar.progress(progress)
                status_text.text(text)
            
            with st.spinner("Загрузка статистики из The Graph..."):
                st.session_state.period_stats = fetch_period_stats_from_graph(
                    period_networks,
                    min_tvl=period_min_tvl,
                    progress_callback=update_progress
                )
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Статистика загружена!")
    
    # Try to load from session state first, then from database
    period_stats = st.session_state.get("period_stats", None)
    
    # If not in session state, try to load from database
    if not period_stats and period_networks:
        try:
            period_stats = load_period_stats_from_db(period_networks, period_min_tvl)
            if period_stats:
                st.session_state.period_stats = period_stats
                st.info("📊 Загружена сохранённая статистика из базы данных")
        except Exception as e:
            logger.debug(f"Could not load period stats from DB: {e}")
    
    # Always show the section, even if no data loaded yet
    if period_stats:
        # Helper function for styling period tables
        def style_period_table(df):
            def highlight_row(row):
                if row["Сеть"] == "ИТОГО":
                    return ["background-color: #1e3a5f; font-weight: bold; color: white"] * len(row)
                return [""] * len(row)
            
            # Get numeric columns (all except "Сеть")
            numeric_cols = [col for col in df.columns if col != "Сеть"]
            format_dict = {col: "{:,}" for col in numeric_cols}
            
            return df.style.apply(highlight_row, axis=1).format(format_dict)
        
        # Positions by period
        st.markdown("#### 📍 Позиции по периодам")
        positions_df = period_stats.get("positions", pd.DataFrame())
        if not positions_df.empty:
            st.dataframe(
                style_period_table(positions_df),
                use_container_width=True,
                hide_index=True,
                height=200
            )
        else:
            st.info("👆 Нажмите «Загрузить статистику» для получения данных")
        
        # Swaps by period
        st.markdown("#### 💱 Свопы по периодам")
        swaps_df = period_stats.get("swaps", pd.DataFrame())
        if not swaps_df.empty:
            st.dataframe(
                style_period_table(swaps_df),
                use_container_width=True,
                hide_index=True,
                height=200
            )
        else:
            st.info("👆 Нажмите «Загрузить статистику» для получения данных")
        
        # Owners by period
        st.markdown("#### 👥 Активные владельцы по периодам")
        st.caption("Уникальные владельцы, создавшие позиции в каждом периоде")
        owners_df = period_stats.get("owners", pd.DataFrame())
        if not owners_df.empty:
            st.dataframe(
                style_period_table(owners_df),
                use_container_width=True,
                hide_index=True,
                height=200
            )
        else:
            st.info("👆 Нажмите «Загрузить статистику» для получения данных")
    else:
        # Show placeholder when no data loaded yet
        st.info("👆 Выберите сети и нажмите «Загрузить статистику» для отображения данных по периодам")
    
    st.markdown("---")
    
    # Load all button
    st.markdown("### 🚀 Загрузить всё (рекомендуется)")
    
    st.markdown("""
    Эта кнопка загрузит:
    1. **Пулы** — ликвидные пулы с выбранных сетей
    2. **Свопы** — сделки за выбранный период для анализа потоков
    3. **Позиции** — LP-позиции за выбранный период для анализа владельцев
    """)
    
    available_networks = [n for n, c in NETWORKS.items() if c.enabled]
    
    # Define period options once for use in both main and manual loading
    period_options = [
        "Последняя неделя",
        "Последний месяц",
        "Последние 3 месяца",
        "Последние 4 месяца",
        "Последний год",
        "Последние 2 года",
        "Последние 3 года",
        "Последние 4 года",
        "Последние 5 лет",
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_networks = st.multiselect(
            "Сети для загрузки",
            available_networks,
            default=["arbitrum", "ethereum"] if "arbitrum" in available_networks else available_networks[:2],
            help="Выберите сети. Arbitrum и Ethereum — основные."
        )
    
    with col2:
        selected_period = st.selectbox(
            "Период данных",
            period_options,
            index=1,  # Default to "Последний месяц"
            help="За какой период загружать свопы и позиции"
        )
        period_hours = get_period_hours(selected_period)
        st.caption(f"({period_hours // 24} дней / {period_hours} часов)")
    
    with col3:
        min_tvl = st.number_input(
            "Мин. TVL пула ($)",
            min_value=10000,
            value=100000,
            step=10000,
            help="Пулы с TVL меньше этого значения не загружаются"
        )
    
    with col4:
        positions_limit = st.number_input(
            "Лимит позиций на сеть",
            min_value=50,
            value=200,
            step=50,
            help="Сколько позиций загружать с каждой сети (больше = дольше)"
        )
    
    if st.button("🚀 Загрузить всё", type="primary", use_container_width=True):
        if not selected_networks:
            st.error("Выберите хотя бы одну сеть")
        elif not GRAPH_API_KEY:
            st.error("GRAPH_API_KEY не настроен. Добавьте его в файл .env")
        else:
            results = load_all_data_action(selected_networks, min_tvl, positions_limit, hours=period_hours)
            
            st.success("✅ Загрузка завершена!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Пулы:**")
                total_pools_loaded = 0
                for net, count in results["pools"].items():
                    if isinstance(count, int):
                        total_pools_loaded += count
                        if count > 0:
                            st.success(f"✅ {net}: {count}")
                        else:
                            st.warning(f"⚠️ {net}: {count} (нет пулов с TVL >= ${min_tvl:,.0f})")
                    else:
                        st.error(f"❌ {net}: {count}")
                
                if results.get("pools_detailed_errors"):
                    with st.expander("🔍 Детали ошибок"):
                        for err in results["pools_detailed_errors"]:
                            st.text(f"  • {err}")
                
                if total_pools_loaded == 0 and not results.get("pools_error"):
                    st.warning("⚠️ Пулы не загружены. Проверьте настройки.")
            with col2:
                st.markdown("**Свопы:**")
                if results.get("swap_warning"):
                    st.warning(f"⚠️ {results['swap_warning']}")
                    if results.get("swap_diagnostic"):
                        diag = results["swap_diagnostic"]
                        if diag.get("total_pools", 0) > 0:
                            st.write(f"Всего пулов в базе: {diag['total_pools']}")
                        if diag.get("pools_in_networks"):
                            st.write(f"Пулов в выбранных сетях: {diag['pools_in_networks']}")
                        if diag.get("available_networks"):
                            st.write(f"Сети в базе: {', '.join(diag['available_networks'])}")
                        if diag.get("requested_networks"):
                            st.write(f"Запрошенные сети: {', '.join(diag['requested_networks'])}")
                        if diag.get("max_tvl"):
                            st.write(f"Макс. TVL в сетях: ${diag['max_tvl']:,.0f}")
                        if diag.get("requested_min_tvl"):
                            st.write(f"Запрошенный мин. TVL: ${diag['requested_min_tvl']:,.0f}")
                        if diag.get("suggestion"):
                            st.info(f"💡 {diag['suggestion']}")
                else:
                    st.write(f"• Загружено: {results['swaps']}")
                    if results.get("total_swap_pools"):
                        st.write(f"• Обработано пулов: {results.get('successful_swap_pools', 0)}/{results['total_swap_pools']}")
                    
                    # Show pools with swaps
                    if results.get("pools_with_swaps"):
                        with st.expander(f"✅ Пулы со свопами ({len(results['pools_with_swaps'])}):"):
                            for pool_info in results["pools_with_swaps"][:10]:
                                st.text(f"  • {pool_info}")
                    
                    # Show pools without swaps
                    if results.get("pools_without_swaps"):
                        with st.expander(f"⚠️ Пулы без свопов ({len(results['pools_without_swaps'])}):"):
                            st.caption("Возможно, в этих пулах нет активности за выбранный период")
                            for pool_info in results["pools_without_swaps"][:10]:
                                st.text(f"  • {pool_info}")
                    
                    if results.get("swap_errors"):
                        st.warning(f"⚠️ Ошибки в {len(results['swap_errors'])} пулах")
                        with st.expander("Детали ошибок"):
                            for err in results["swap_errors"][:5]:
                                st.text(err)
            with col3:
                st.markdown("**Позиции:**")
                for net, data in results["positions"].items():
                    if isinstance(data, dict):
                        st.write(f"• {net}: {data.get('open', 0)} откр. + {data.get('closed', 0)} закр.")
                    else:
                        st.write(f"• {net}: {data}")
    
    st.markdown("---")
    
    # Manual loading options
    with st.expander("⚙️ Ручная загрузка (для продвинутых)"):
        tab1, tab2, tab3 = st.tabs(["Пулы", "Свопы", "Позиции"])
        
        with tab1:
            st.markdown("Загрузить только пулы:")
            if st.button("Загрузить пулы"):
                loader = PoolLoader()
                total_loaded = 0
                errors = []
                
                with st.spinner("Загрузка пулов..."):
                    for net in selected_networks:
                        try:
                            with session_scope() as session:
                                count = loader.load_pools_for_network(session, net, min_tvl=min_tvl)
                                total_loaded += count
                                
                                # Verify pools were saved
                                verify = session.query(Pool).filter(
                                    Pool.network == net,
                                    Pool.tvl_usd >= min_tvl
                                ).count()
                                
                                if verify == 0 and count > 0:
                                    errors.append(f"{net}: загружено {count}, но не сохранено в БД")
                                elif verify > 0:
                                    st.info(f"✅ {net}: загружено и сохранено {verify} пулов")
                        except Exception as e:
                            error_msg = f"{net}: {str(e)[:50]}"
                            errors.append(error_msg)
                            logger.error(f"Error loading pools from {net}: {e}", exc_info=True)
                
                if errors:
                    st.warning(f"⚠️ Загружено {total_loaded} пулов, но были ошибки:")
                    for err in errors:
                        st.text(f"  • {err}")
                elif total_loaded > 0:
                    # Verify final count
                    with session_scope() as session:
                        final_count = session.query(Pool).filter(
                            Pool.network.in_(selected_networks),
                            Pool.tvl_usd >= min_tvl
                        ).count()
                        st.success(f"✅ Загружено и сохранено {final_count} пулов!")
                else:
                    st.error("❌ Не удалось загрузить пулы. Проверьте GRAPH_API_KEY и настройки.")
        
        with tab2:
            st.markdown("Загрузить только свопы:")
            manual_period = st.selectbox(
                "Период",
                period_options,
                index=1,
                key="manual_swaps_period"
            )
            manual_swaps_hours = get_period_hours(manual_period)
            manual_swaps_networks = st.multiselect(
                "Сети",
                available_networks,
                default=selected_networks if selected_networks else [],
                key="manual_swaps_networks"
            )
            test_mode = st.checkbox("Тестовый режим (показать детали)", key="test_swaps_mode")
            
            if st.button("Загрузить свопы"):
                if not manual_swaps_networks:
                    st.error("Выберите хотя бы одну сеть")
                else:
                    with st.spinner("Загрузка свопов..."):
                        total_swaps = 0
                        errors = []
                        pools_with_swaps = []
                        pools_without_swaps = []
                        
                        with session_scope() as session:
                            pools = session.query(Pool).filter(
                                Pool.tvl_usd >= min_tvl,
                                Pool.network.in_(manual_swaps_networks)
                            ).order_by(Pool.tvl_usd.desc()).limit(30).all()
                            
                            if not pools:
                                # Check what's in the database
                                total_pools = session.query(Pool).count()
                                pools_in_selected = session.query(Pool).filter(
                                    Pool.network.in_(manual_swaps_networks)
                                ).count()
                                pools_above_tvl = session.query(Pool).filter(
                                    Pool.tvl_usd >= min_tvl,
                                    Pool.network.in_(manual_swaps_networks)
                                ).count()
                                
                                st.warning("⚠️ Нет пулов для загрузки свопов")
                                
                                if total_pools == 0:
                                    st.error("❌ В базе данных нет пулов! Перейдите на вкладку 'Пулы' и загрузите их.")
                                elif pools_in_selected == 0:
                                    st.warning(f"В базе нет пулов для сетей: {', '.join(manual_swaps_networks)}")
                                    st.info(f"Всего пулов в базе: {total_pools}")
                                elif pools_above_tvl == 0:
                                    max_tvl = session.query(Pool.tvl_usd).filter(
                                        Pool.network.in_(manual_swaps_networks)
                                    ).order_by(Pool.tvl_usd.desc()).first()
                                    max_tvl_val = float(max_tvl[0]) if max_tvl and max_tvl[0] else 0
                                    st.warning(f"Нет пулов с TVL >= ${min_tvl:,.0f}")
                                    st.info(f"Максимальный TVL: ${max_tvl_val:,.0f} | Пулов в сетях: {pools_in_selected}")
                                    st.caption("💡 Попробуйте уменьшить 'Мин. TVL пула'")
                            else:
                                st.info(f"🔍 Проверяю {len(pools)} пулов за период {manual_swaps_hours // 24} дней...")
                                loader = SwapLoader()
                                
                                for pool in pools:
                                    try:
                                        count = loader.load_swaps_for_pool(
                                            session, pool, hours=manual_swaps_hours, limit=50
                                        )
                                        total_swaps += count
                                        if count > 0:
                                            pools_with_swaps.append(f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol} ({count} свопов)")
                                        else:
                                            pools_without_swaps.append(f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol} (адрес: {pool.address[:10]}...)")
                                    except Exception as e:
                                        error_msg = f"{pool.network}/{pool.token0_symbol}-{pool.token1_symbol}: {str(e)[:50]}"
                                        errors.append(error_msg)
                                        logger.error(f"Error loading swaps for pool {pool.address}: {e}", exc_info=True)
                                
                                # Show results
                                if total_swaps > 0:
                                    st.success(f"✅ Загружено {total_swaps} свопов из {len(pools_with_swaps)} пулов!")
                                    if pools_with_swaps:
                                        with st.expander(f"✅ Пулы со свопами ({len(pools_with_swaps)}):"):
                                            for pool_info in pools_with_swaps:
                                                st.text(f"  • {pool_info}")
                                else:
                                    st.warning(f"⚠️ Свопы не найдены в {len(pools)} пулах за выбранный период")
                                    st.caption("💡 Попробуйте:")
                                    st.caption("  • Увеличить период (например, 'Последний месяц')")
                                    st.caption("  • Проверить, что пулы имеют активность")
                                    st.caption("  • Убедиться, что пулы загружены правильно")
                                
                                if pools_without_swaps and test_mode:
                                    with st.expander(f"🔍 Пулы без свопов ({len(pools_without_swaps)}):"):
                                        for pool_info in pools_without_swaps[:20]:
                                            st.text(f"  • {pool_info}")
                                
                                if errors:
                                    st.warning(f"⚠️ Ошибки в {len(errors)} пулах")
                                    with st.expander("Детали ошибок"):
                                        for err in errors[:10]:
                                            st.text(err)
        
        with tab3:
            st.markdown("Загрузить только позиции:")
            manual_pos_period = st.selectbox(
                "Период",
                period_options,
                index=1,
                key="manual_positions_period"
            )
            manual_pos_hours = get_period_hours(manual_pos_period)
            manual_networks = st.multiselect(
                "Сети",
                available_networks,
                default=selected_networks if selected_networks else [],
                key="manual_positions_networks"
            )
            if st.button("Загрузить позиции"):
                if not manual_networks:
                    st.error("Выберите хотя бы одну сеть")
                else:
                    loader = PositionLoader()
                    with st.spinner("Загрузка позиций..."):
                        with session_scope() as session:
                            for network in manual_networks:
                                loader.load_positions_from_events(
                                    session, network, min_amount_usd="100", 
                                    limit=positions_limit, hours=manual_pos_hours
                                )
                        calculate_positions_usd()
                    st.success("Позиции загружены!")


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
