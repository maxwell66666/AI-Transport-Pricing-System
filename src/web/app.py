"""
Web应用模块

此模块提供基于Dash的Web界面，用于展示AI运输报价系统的功能。
"""

import os
import sys
import logging
import requests
import json
from pathlib import Path
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# 将项目根目录添加到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API基础URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# 创建Dash应用
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AI运输报价系统",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

# 设置服务器
server = app.server

# 定义布局
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("AI运输报价系统", className="text-center my-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("运输信息"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("起始地点"),
                            dcc.Dropdown(
                                id="origin-dropdown",
                                placeholder="选择起始地点",
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("目的地点"),
                            dcc.Dropdown(
                                id="destination-dropdown",
                                placeholder="选择目的地点",
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("运输方式"),
                            dcc.Dropdown(
                                id="transport-mode-dropdown",
                                placeholder="选择运输方式",
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("货物类型"),
                            dcc.Dropdown(
                                id="cargo-type-dropdown",
                                placeholder="选择货物类型",
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("重量 (kg)"),
                            dbc.Input(
                                id="weight-input",
                                type="number",
                                placeholder="输入重量",
                                min=0,
                                step=0.1,
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("体积 (m³)"),
                            dbc.Input(
                                id="volume-input",
                                type="number",
                                placeholder="输入体积",
                                min=0,
                                step=0.01,
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("距离 (km) [可选]"),
                            dbc.Input(
                                id="distance-input",
                                type="number",
                                placeholder="输入距离 (可选)",
                                min=0,
                                step=0.1,
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("运输时间 (天) [可选]"),
                            dbc.Input(
                                id="transit-time-input",
                                type="number",
                                placeholder="输入运输时间 (可选)",
                                min=0,
                                step=1,
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "预测报价",
                                id="predict-button",
                                color="primary",
                                className="w-100",
                            ),
                        ], width=12),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("报价结果"),
                dbc.CardBody([
                    html.Div(id="prediction-result", className="mb-4"),
                    dcc.Graph(id="price-breakdown-chart"),
                ]),
            ], className="mb-4"),
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("历史报价趋势"),
                dbc.CardBody([
                    dcc.Graph(id="historical-trend-chart"),
                ]),
            ]),
        ], width=12),
    ]),
    
    dbc.Modal([
        dbc.ModalHeader("错误"),
        dbc.ModalBody(id="error-modal-body"),
        dbc.ModalFooter(
            dbc.Button("关闭", id="close-error-modal", className="ml-auto")
        ),
    ], id="error-modal"),
    
    # 存储组件
    dcc.Store(id="locations-store"),
    dcc.Store(id="transport-modes-store"),
    dcc.Store(id="cargo-types-store"),
    dcc.Store(id="prediction-store"),
    
], fluid=True)


# 回调函数：加载地点数据
@app.callback(
    Output("locations-store", "data"),
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True,
)
def load_locations(n_clicks):
    """加载地点数据"""
    try:
        response = requests.get(f"{API_BASE_URL}/quotes/locations")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"加载地点数据时出错: {e}")
        return []


# 回调函数：加载运输方式数据
@app.callback(
    Output("transport-modes-store", "data"),
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True,
)
def load_transport_modes(n_clicks):
    """加载运输方式数据"""
    try:
        response = requests.get(f"{API_BASE_URL}/quotes/transport-modes")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"加载运输方式数据时出错: {e}")
        return []


# 回调函数：加载货物类型数据
@app.callback(
    Output("cargo-types-store", "data"),
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True,
)
def load_cargo_types(n_clicks):
    """加载货物类型数据"""
    try:
        response = requests.get(f"{API_BASE_URL}/quotes/cargo-types")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"加载货物类型数据时出错: {e}")
        return []


# 回调函数：更新地点下拉框
@app.callback(
    [Output("origin-dropdown", "options"), Output("destination-dropdown", "options")],
    Input("locations-store", "data"),
)
def update_location_dropdowns(locations):
    """更新地点下拉框"""
    if not locations:
        return [], []
    
    options = [{"label": f"{loc['name']} ({loc['country']})", "value": loc['id']} for loc in locations]
    return options, options


# 回调函数：更新运输方式下拉框
@app.callback(
    Output("transport-mode-dropdown", "options"),
    Input("transport-modes-store", "data"),
)
def update_transport_mode_dropdown(transport_modes):
    """更新运输方式下拉框"""
    if not transport_modes:
        return []
    
    options = [{"label": mode['name'], "value": mode['id']} for mode in transport_modes]
    return options


# 回调函数：更新货物类型下拉框
@app.callback(
    Output("cargo-type-dropdown", "options"),
    Input("cargo-types-store", "data"),
)
def update_cargo_type_dropdown(cargo_types):
    """更新货物类型下拉框"""
    if not cargo_types:
        return []
    
    options = [{"label": cargo['name'], "value": cargo['id']} for cargo in cargo_types]
    return options


# 回调函数：预测报价
@app.callback(
    [
        Output("prediction-store", "data"),
        Output("error-modal", "is_open"),
        Output("error-modal-body", "children"),
    ],
    Input("predict-button", "n_clicks"),
    [
        State("origin-dropdown", "value"),
        State("destination-dropdown", "value"),
        State("transport-mode-dropdown", "value"),
        State("cargo-type-dropdown", "value"),
        State("weight-input", "value"),
        State("volume-input", "value"),
        State("distance-input", "value"),
        State("transit-time-input", "value"),
    ],
    prevent_initial_call=True,
)
def predict_quote(n_clicks, origin_id, destination_id, transport_mode_id, cargo_type_id, 
                  weight, volume, distance, transit_time):
    """预测报价"""
    # 验证输入
    if not all([origin_id, destination_id, transport_mode_id, cargo_type_id, weight, volume]):
        return None, True, "请填写所有必填字段（起始地点、目的地点、运输方式、货物类型、重量、体积）"
    
    try:
        # 准备请求数据
        data = {
            "origin_location_id": origin_id,
            "destination_location_id": destination_id,
            "transport_mode_id": transport_mode_id,
            "cargo_type_id": cargo_type_id,
            "weight": float(weight),
            "volume": float(volume),
        }
        
        # 添加可选字段
        if distance:
            data["distance"] = float(distance)
        if transit_time:
            data["typical_transit_time"] = int(transit_time)
        
        # 发送请求
        response = requests.post(f"{API_BASE_URL}/quotes/predict", json=data)
        response.raise_for_status()
        
        # 返回结果
        return response.json(), False, ""
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求错误: {e}")
        error_message = "API请求错误，请检查API服务是否正常运行"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                if 'detail' in error_data:
                    error_message = f"API错误: {error_data['detail']}"
            except:
                error_message = f"API错误: {e}"
        return None, True, error_message
    
    except Exception as e:
        logger.error(f"预测报价时出错: {e}")
        return None, True, f"预测报价时出错: {str(e)}"


# 回调函数：更新报价结果
@app.callback(
    Output("prediction-result", "children"),
    Input("prediction-store", "data"),
)
def update_prediction_result(prediction_data):
    """更新报价结果"""
    if not prediction_data:
        return html.Div("请填写运输信息并点击"预测报价"按钮", className="text-muted")
    
    # 准备显示内容
    content = [
        html.H3(f"预测价格: ${prediction_data['predicted_price']:.2f} {prediction_data['currency']}", className="text-success"),
        html.P(f"使用模型: {prediction_data['model_used']}"),
    ]
    
    # 如果有置信度，添加到内容中
    if prediction_data.get('confidence'):
        content.append(html.P(f"预测置信度: {prediction_data['confidence'] * 100:.1f}%"))
    
    return html.Div(content)


# 回调函数：更新价格明细图表
@app.callback(
    Output("price-breakdown-chart", "figure"),
    Input("prediction-store", "data"),
)
def update_price_breakdown_chart(prediction_data):
    """更新价格明细图表"""
    if not prediction_data or not prediction_data.get('price_breakdown'):
        # 返回空图表
        return go.Figure().update_layout(
            title="价格明细",
            xaxis_title="费用类型",
            yaxis_title="金额 (USD)",
        )
    
    # 准备数据
    breakdown = prediction_data['price_breakdown']
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    
    # 创建饼图
    fig = px.pie(
        names=labels,
        values=values,
        title="价格明细",
        labels={'names': '费用类型', 'values': '金额 (USD)'},
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title="费用类型",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    
    return fig


# 回调函数：更新历史趋势图表
@app.callback(
    Output("historical-trend-chart", "figure"),
    Input("prediction-store", "data"),
)
def update_historical_trend_chart(prediction_data):
    """更新历史趋势图表"""
    # 创建模拟历史数据
    # 在实际应用中，这应该从数据库中获取
    if not prediction_data:
        # 返回空图表
        return go.Figure().update_layout(
            title="历史报价趋势",
            xaxis_title="月份",
            yaxis_title="平均价格 (USD)",
        )
    
    # 生成过去12个月的模拟数据
    import datetime
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year
    
    months = []
    for i in range(11, -1, -1):
        month = (current_month - i) % 12
        if month == 0:
            month = 12
        year = current_year - ((current_month - i) // 12)
        months.append(f"{year}-{month:02d}")
    
    # 基于预测价格生成模拟历史数据
    base_price = prediction_data['predicted_price']
    np.random.seed(42)  # 固定随机种子，使结果可重现
    
    # 生成一个有季节性和轻微上升趋势的价格序列
    seasonal_factor = np.sin(np.linspace(0, 2*np.pi, 12)) * 0.15 + 1  # 季节性因子
    trend_factor = np.linspace(0.9, 1.1, 12)  # 趋势因子
    random_factor = np.random.normal(1, 0.05, 12)  # 随机因子
    
    prices = base_price * seasonal_factor * trend_factor * random_factor
    
    # 创建数据框
    df = pd.DataFrame({
        'month': months,
        'price': prices
    })
    
    # 创建折线图
    fig = px.line(
        df,
        x='month',
        y='price',
        title="历史报价趋势 (模拟数据)",
        labels={'month': '月份', 'price': '平均价格 (USD)'},
        markers=True,
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(t=50, b=50, l=50, r=20),
    )
    
    # 添加当前预测价格的水平线
    fig.add_hline(
        y=base_price,
        line_dash="dash",
        line_color="red",
        annotation_text="当前预测价格",
        annotation_position="bottom right",
    )
    
    return fig


# 回调函数：关闭错误模态框
@app.callback(
    Output("error-modal", "is_open", allow_duplicate=True),
    Input("close-error-modal", "n_clicks"),
    prevent_initial_call=True,
)
def close_error_modal(n_clicks):
    """关闭错误模态框"""
    return False


# 运行应用
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050) 