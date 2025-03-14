#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
初始化数据库脚本

此脚本用于创建数据库表结构和加载初始数据。
可以在开发环境或生产环境中使用，用于初始化数据库或重置数据库。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy import text

from src.db.database import get_engine, init_db
from src.db.models import Base, CargoType, Location, TransportMode, User
from src.utils.config import settings
from src.utils.logging import setup_logging

# 设置日志
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="初始化数据库")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置数据库（删除所有表并重新创建）",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="加载示例数据",
    )
    parser.add_argument(
        "--env",
        choices=["development", "testing", "production"],
        default=settings.ENVIRONMENT,
        help=f"运行环境 (默认: {settings.ENVIRONMENT})",
    )
    return parser.parse_args()


def reset_database(engine):
    """删除所有表并重新创建"""
    logger.info("正在重置数据库...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("数据库重置完成")


def create_tables(engine):
    """创建所有表"""
    logger.info("正在创建数据库表...")
    Base.metadata.create_all(engine)
    logger.info("数据库表创建完成")


def create_initial_data(engine):
    """创建初始数据"""
    from sqlalchemy.orm import Session

    logger.info("正在创建初始数据...")
    
    with Session(engine) as session:
        # 检查是否已存在管理员用户
        admin_exists = session.query(User).filter(User.username == "admin").first()
        if not admin_exists:
            # 创建管理员用户
            admin_user = User(
                username="admin",
                email="admin@example.com",
                full_name="系统管理员",
                is_active=True,
                is_superuser=True,
            )
            admin_user.set_password("admin123")  # 在生产环境中应使用强密码
            session.add(admin_user)
            logger.info("已创建管理员用户")
        
        # 检查是否已存在运输方式
        transport_modes_exist = session.query(TransportMode).first()
        if not transport_modes_exist:
            # 创建基本运输方式
            transport_modes = [
                TransportMode(name="公路运输", code="ROAD", description="通过卡车和货车进行的陆路运输"),
                TransportMode(name="铁路运输", code="RAIL", description="通过铁路网络进行的运输"),
                TransportMode(name="海运", code="SEA", description="通过船舶进行的海上运输"),
                TransportMode(name="空运", code="AIR", description="通过飞机进行的航空运输"),
                TransportMode(name="多式联运", code="MULTIMODAL", description="结合多种运输方式的综合运输"),
                TransportMode(name="快递", code="EXPRESS", description="快速的包裹递送服务"),
            ]
            session.add_all(transport_modes)
            logger.info("已创建基本运输方式")
        
        # 检查是否已存在货物类型
        cargo_types_exist = session.query(CargoType).first()
        if not cargo_types_exist:
            # 创建基本货物类型
            cargo_types = [
                CargoType(name="普通货物", code="GENERAL", description="无特殊要求的一般货物"),
                CargoType(name="易碎品", code="FRAGILE", description="需要特殊处理的易碎货物"),
                CargoType(name="危险品", code="DANGEROUS", description="符合危险品运输规定的货物"),
                CargoType(name="冷藏品", code="REFRIGERATED", description="需要温控的冷藏货物"),
                CargoType(name="大型货物", code="OVERSIZED", description="超出标准尺寸的大型货物"),
                CargoType(name="贵重物品", code="VALUABLE", description="高价值需要特殊保护的货物"),
                CargoType(name="活体动物", code="LIVE_ANIMALS", description="活体动物运输"),
                CargoType(name="医药产品", code="PHARMACEUTICAL", description="医药产品，可能需要特殊处理"),
            ]
            session.add_all(cargo_types)
            logger.info("已创建基本货物类型")
        
        # 检查是否已存在位置信息
        locations_exist = session.query(Location).first()
        if not locations_exist:
            # 创建基本位置信息
            locations = [
                Location(name="北京", code="BJ", country="中国", region="华北", city="北京", 
                         address="北京市", postal_code="100000", latitude=39.9042, longitude=116.4074),
                Location(name="上海", code="SH", country="中国", region="华东", city="上海", 
                         address="上海市", postal_code="200000", latitude=31.2304, longitude=121.4737),
                Location(name="广州", code="GZ", country="中国", region="华南", city="广州", 
                         address="广州市", postal_code="510000", latitude=23.1291, longitude=113.2644),
                Location(name="深圳", code="SZ", country="中国", region="华南", city="深圳", 
                         address="深圳市", postal_code="518000", latitude=22.5431, longitude=114.0579),
                Location(name="成都", code="CD", country="中国", region="西南", city="成都", 
                         address="成都市", postal_code="610000", latitude=30.5728, longitude=104.0668),
                Location(name="武汉", code="WH", country="中国", region="华中", city="武汉", 
                         address="武汉市", postal_code="430000", latitude=30.5928, longitude=114.3055),
                Location(name="西安", code="XA", country="中国", region="西北", city="西安", 
                         address="西安市", postal_code="710000", latitude=34.3416, longitude=108.9398),
                Location(name="重庆", code="CQ", country="中国", region="西南", city="重庆", 
                         address="重庆市", postal_code="400000", latitude=29.4316, longitude=106.9123),
                Location(name="青岛", code="QD", country="中国", region="华东", city="青岛", 
                         address="青岛市", postal_code="266000", latitude=36.0671, longitude=120.3826),
                Location(name="杭州", code="HZ", country="中国", region="华东", city="杭州", 
                         address="杭州市", postal_code="310000", latitude=30.2741, longitude=120.1551),
            ]
            session.add_all(locations)
            logger.info("已创建基本位置信息")
        
        session.commit()
    
    logger.info("初始数据创建完成")


def load_sample_data():
    """加载示例数据"""
    logger.info("正在加载示例数据...")
    # 调用示例数据加载脚本
    try:
        from scripts.load_sample_data import main as load_sample_data_main
        load_sample_data_main()
        logger.info("示例数据加载完成")
    except ImportError:
        logger.error("无法导入示例数据加载脚本")
    except Exception as e:
        logger.error(f"加载示例数据时出错: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置环境变量
    os.environ["ENVIRONMENT"] = args.env
    
    # 设置日志
    setup_logging()
    
    logger.info(f"正在初始化数据库，环境: {args.env}")
    
    try:
        # 获取数据库引擎
        engine = get_engine()
        
        # 初始化数据库连接
        init_db()
        
        # 根据参数决定是重置还是创建
        if args.reset:
            reset_database(engine)
        else:
            create_tables(engine)
        
        # 创建初始数据
        create_initial_data(engine)
        
        # 加载示例数据
        if args.sample_data:
            load_sample_data()
        
        logger.info("数据库初始化完成")
        
    except Exception as e:
        logger.error(f"初始化数据库时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 