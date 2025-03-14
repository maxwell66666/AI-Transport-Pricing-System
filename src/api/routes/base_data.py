"""
基础数据路由处理器

本模块提供运输方式、货物类型、位置等基础数据的API路由。
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.auth import get_current_active_user
from src.db.database import get_db
from src.db.models import User, TransportMode, CargoType, Location, DistanceMatrix
from src.db.repository import (
    TransportModeRepository, CargoTypeRepository,
    LocationRepository, DistanceMatrixRepository
)
from src.utils.logging import get_logger

# 创建路由器
router = APIRouter(prefix="/base-data", tags=["基础数据"])

# 日志记录器
logger = get_logger(__name__)


# 运输方式模型
class TransportModeResponse(BaseModel):
    """运输方式响应模型"""
    id: int
    name: str
    description: Optional[str] = None
    is_active: bool


# 货物类型模型
class CargoTypeResponse(BaseModel):
    """货物类型响应模型"""
    id: int
    name: str
    description: Optional[str] = None
    is_dangerous: bool
    requires_temperature_control: bool


# 位置模型
class LocationResponse(BaseModel):
    """位置响应模型"""
    id: int
    name: str
    country: str
    city: str
    address: Optional[str] = None
    postal_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_port: bool
    is_airport: bool


# 距离矩阵模型
class DistanceMatrixResponse(BaseModel):
    """距离矩阵响应模型"""
    id: int
    origin_location_id: int
    destination_location_id: int
    transport_mode_id: int
    distance: float
    typical_transit_time: float
    origin_location: LocationResponse
    destination_location: LocationResponse
    transport_mode: TransportModeResponse


@router.get("/transport-modes", response_model=List[TransportModeResponse])
async def get_transport_modes(
    active_only: bool = Query(True, description="是否只返回活跃的运输方式"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取运输方式列表
    
    Args:
        active_only: 是否只返回活跃的运输方式
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        运输方式列表
    """
    transport_mode_repo = TransportModeRepository(db)
    
    if active_only:
        transport_modes = transport_mode_repo.get_active()
    else:
        transport_modes = transport_mode_repo.get_all()
    
    return [
        {
            "id": mode.id,
            "name": mode.name,
            "description": mode.description,
            "is_active": mode.is_active
        }
        for mode in transport_modes
    ]


@router.get("/transport-modes/{mode_id}", response_model=TransportModeResponse)
async def get_transport_mode(
    mode_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取运输方式详情
    
    Args:
        mode_id: 运输方式ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        运输方式详情
        
    Raises:
        HTTPException: 运输方式不存在
    """
    transport_mode_repo = TransportModeRepository(db)
    mode = transport_mode_repo.get_by_id(mode_id)
    
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"运输方式ID {mode_id} 不存在"
        )
    
    return {
        "id": mode.id,
        "name": mode.name,
        "description": mode.description,
        "is_active": mode.is_active
    }


@router.get("/cargo-types", response_model=List[CargoTypeResponse])
async def get_cargo_types(
    dangerous_only: bool = Query(False, description="是否只返回危险货物类型"),
    temperature_controlled_only: bool = Query(False, description="是否只返回需要温控的货物类型"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取货物类型列表
    
    Args:
        dangerous_only: 是否只返回危险货物类型
        temperature_controlled_only: 是否只返回需要温控的货物类型
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        货物类型列表
    """
    cargo_type_repo = CargoTypeRepository(db)
    
    if dangerous_only:
        cargo_types = cargo_type_repo.get_dangerous()
    elif temperature_controlled_only:
        cargo_types = cargo_type_repo.get_temperature_controlled()
    else:
        cargo_types = cargo_type_repo.get_all()
    
    return [
        {
            "id": cargo_type.id,
            "name": cargo_type.name,
            "description": cargo_type.description,
            "is_dangerous": cargo_type.is_dangerous,
            "requires_temperature_control": cargo_type.requires_temperature_control
        }
        for cargo_type in cargo_types
    ]


@router.get("/cargo-types/{cargo_type_id}", response_model=CargoTypeResponse)
async def get_cargo_type(
    cargo_type_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取货物类型详情
    
    Args:
        cargo_type_id: 货物类型ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        货物类型详情
        
    Raises:
        HTTPException: 货物类型不存在
    """
    cargo_type_repo = CargoTypeRepository(db)
    cargo_type = cargo_type_repo.get_by_id(cargo_type_id)
    
    if not cargo_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"货物类型ID {cargo_type_id} 不存在"
        )
    
    return {
        "id": cargo_type.id,
        "name": cargo_type.name,
        "description": cargo_type.description,
        "is_dangerous": cargo_type.is_dangerous,
        "requires_temperature_control": cargo_type.requires_temperature_control
    }


@router.get("/locations", response_model=List[LocationResponse])
async def get_locations(
    country: Optional[str] = Query(None, description="按国家筛选"),
    city: Optional[str] = Query(None, description="按城市筛选"),
    is_port: Optional[bool] = Query(None, description="是否为港口"),
    is_airport: Optional[bool] = Query(None, description="是否为机场"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    skip: int = Query(0, description="跳过的记录数"),
    limit: int = Query(100, description="返回的记录数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取位置列表
    
    Args:
        country: 按国家筛选
        city: 按城市筛选
        is_port: 是否为港口
        is_airport: 是否为机场
        search: 搜索关键词
        skip: 跳过的记录数
        limit: 返回的记录数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        位置列表
    """
    location_repo = LocationRepository(db)
    
    # 构建过滤条件
    filters = {}
    
    if country:
        filters["country"] = country
    
    if city:
        filters["city"] = city
    
    if is_port is not None:
        filters["is_port"] = is_port
    
    if is_airport is not None:
        filters["is_airport"] = is_airport
    
    # 如果有搜索关键词，使用搜索方法
    if search:
        locations = location_repo.search(db, query=search)
    else:
        # 否则使用过滤条件
        locations = location_repo.get_filtered(
            skip=skip,
            limit=limit,
            **filters
        )
    
    return [
        {
            "id": location.id,
            "name": location.name,
            "country": location.country,
            "city": location.city,
            "address": location.address,
            "postal_code": location.postal_code,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "is_port": location.is_port,
            "is_airport": location.is_airport
        }
        for location in locations
    ]


@router.get("/locations/{location_id}", response_model=LocationResponse)
async def get_location(
    location_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取位置详情
    
    Args:
        location_id: 位置ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        位置详情
        
    Raises:
        HTTPException: 位置不存在
    """
    location_repo = LocationRepository(db)
    location = location_repo.get_by_id(location_id)
    
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"位置ID {location_id} 不存在"
        )
    
    return {
        "id": location.id,
        "name": location.name,
        "country": location.country,
        "city": location.city,
        "address": location.address,
        "postal_code": location.postal_code,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "is_port": location.is_port,
        "is_airport": location.is_airport
    }


@router.get("/distance-matrix", response_model=List[DistanceMatrixResponse])
async def get_distance_matrix(
    origin_id: Optional[int] = Query(None, description="起始地点ID"),
    destination_id: Optional[int] = Query(None, description="目的地点ID"),
    transport_mode_id: Optional[int] = Query(None, description="运输方式ID"),
    skip: int = Query(0, description="跳过的记录数"),
    limit: int = Query(100, description="返回的记录数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取距离矩阵列表
    
    Args:
        origin_id: 起始地点ID
        destination_id: 目的地点ID
        transport_mode_id: 运输方式ID
        skip: 跳过的记录数
        limit: 返回的记录数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        距离矩阵列表
    """
    distance_matrix_repo = DistanceMatrixRepository(db)
    
    # 构建过滤条件
    filters = {}
    
    if origin_id:
        filters["origin_location_id"] = origin_id
    
    if destination_id:
        filters["destination_location_id"] = destination_id
    
    if transport_mode_id:
        filters["transport_mode_id"] = transport_mode_id
    
    # 获取距离矩阵列表
    distance_matrices = distance_matrix_repo.get_filtered(
        skip=skip,
        limit=limit,
        **filters
    )
    
    # 预加载关联数据
    location_repo = LocationRepository(db)
    transport_mode_repo = TransportModeRepository(db)
    
    # 构建响应
    result = []
    for dm in distance_matrices:
        origin = location_repo.get_by_id(dm.origin_location_id)
        destination = location_repo.get_by_id(dm.destination_location_id)
        transport_mode = transport_mode_repo.get_by_id(dm.transport_mode_id)
        
        result.append({
            "id": dm.id,
            "origin_location_id": dm.origin_location_id,
            "destination_location_id": dm.destination_location_id,
            "transport_mode_id": dm.transport_mode_id,
            "distance": dm.distance,
            "typical_transit_time": dm.typical_transit_time,
            "origin_location": {
                "id": origin.id,
                "name": origin.name,
                "country": origin.country,
                "city": origin.city,
                "address": origin.address,
                "postal_code": origin.postal_code,
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "is_port": origin.is_port,
                "is_airport": origin.is_airport
            },
            "destination_location": {
                "id": destination.id,
                "name": destination.name,
                "country": destination.country,
                "city": destination.city,
                "address": destination.address,
                "postal_code": destination.postal_code,
                "latitude": destination.latitude,
                "longitude": destination.longitude,
                "is_port": destination.is_port,
                "is_airport": destination.is_airport
            },
            "transport_mode": {
                "id": transport_mode.id,
                "name": transport_mode.name,
                "description": transport_mode.description,
                "is_active": transport_mode.is_active
            }
        })
    
    return result


@router.get("/distance-matrix/{distance_matrix_id}", response_model=DistanceMatrixResponse)
async def get_distance_matrix_by_id(
    distance_matrix_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取距离矩阵详情
    
    Args:
        distance_matrix_id: 距离矩阵ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        距离矩阵详情
        
    Raises:
        HTTPException: 距离矩阵不存在
    """
    distance_matrix_repo = DistanceMatrixRepository(db)
    dm = distance_matrix_repo.get_by_id(distance_matrix_id)
    
    if not dm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"距离矩阵ID {distance_matrix_id} 不存在"
        )
    
    # 获取关联数据
    location_repo = LocationRepository(db)
    transport_mode_repo = TransportModeRepository(db)
    
    origin = location_repo.get_by_id(dm.origin_location_id)
    destination = location_repo.get_by_id(dm.destination_location_id)
    transport_mode = transport_mode_repo.get_by_id(dm.transport_mode_id)
    
    return {
        "id": dm.id,
        "origin_location_id": dm.origin_location_id,
        "destination_location_id": dm.destination_location_id,
        "transport_mode_id": dm.transport_mode_id,
        "distance": dm.distance,
        "typical_transit_time": dm.typical_transit_time,
        "origin_location": {
            "id": origin.id,
            "name": origin.name,
            "country": origin.country,
            "city": origin.city,
            "address": origin.address,
            "postal_code": origin.postal_code,
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "is_port": origin.is_port,
            "is_airport": origin.is_airport
        },
        "destination_location": {
            "id": destination.id,
            "name": destination.name,
            "country": destination.country,
            "city": destination.city,
            "address": destination.address,
            "postal_code": destination.postal_code,
            "latitude": destination.latitude,
            "longitude": destination.longitude,
            "is_port": destination.is_port,
            "is_airport": destination.is_airport
        },
        "transport_mode": {
            "id": transport_mode.id,
            "name": transport_mode.name,
            "description": transport_mode.description,
            "is_active": transport_mode.is_active
        }
    }


@router.get("/distance", response_model=Dict[str, Any])
async def get_distance(
    origin_id: int = Query(..., description="起始地点ID"),
    destination_id: int = Query(..., description="目的地点ID"),
    transport_mode_id: int = Query(..., description="运输方式ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取两地之间的距离和运输时间
    
    Args:
        origin_id: 起始地点ID
        destination_id: 目的地点ID
        transport_mode_id: 运输方式ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        距离和运输时间
        
    Raises:
        HTTPException: 距离矩阵不存在
    """
    distance_matrix_repo = DistanceMatrixRepository(db)
    dm = distance_matrix_repo.get_by_locations_and_mode(
        origin_id=origin_id,
        destination_id=destination_id,
        mode_id=transport_mode_id
    )
    
    if not dm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到从 {origin_id} 到 {destination_id} 使用运输方式 {transport_mode_id} 的距离数据"
        )
    
    return {
        "origin_id": dm.origin_location_id,
        "destination_id": dm.destination_location_id,
        "transport_mode_id": dm.transport_mode_id,
        "distance": dm.distance,
        "typical_transit_time": dm.typical_transit_time
    } 