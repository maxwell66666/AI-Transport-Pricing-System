"""
Web仪表盘模块

使用Dash创建一个基于Web的报表界面，用于展示运输报价系统的功能。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# 设置API基础URL
API_BASE_URL = "http://localhost:8000"

# 创建Dash应用
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AI运输报价系统",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

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
                dbc.CardHeader("价格预测", className="text-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("起始地点"),
                            dcc.Dropdown(id="origin-dropdown", placeholder="选择起始地点"),
                        ], width=6),
                        dbc.Col([
                            html.Label("目的地点"),
                            dcc.Dropdown(id="destination-dropdown", placeholder="选择目的地点"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("运输方式"),
                            dcc.Dropdown(id="transport-mode-dropdown", placeholder="选择运输方式"),
                        ], width=6),
                        dbc.Col([
                            html.Label("货物类型"),
                            dcc.Dropdown(id="cargo-type-dropdown", placeholder="选择货物类型"),
                        ], width=6),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("重量 (kg)"),
                            dbc.Input(id="weight-input", type="number", min=0, step=0.1, value=100),
                        ], width=6),
                        dbc.Col([
                            html.Label("体积 (m³)"),
                            dbc.Input(id="volume-input", type="number", min=0, step=0.1, value=1),
                        ], width=6),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("特殊要求"),
                            dbc.Checklist(
                                id="special-requirements-checklist",
                                options=[
                                    {"label": "易碎物品", "value": "fragile"},
                                    {"label": "温度控制", "value": "temperature_control"},
                                    {"label": "定制包装", "value": "custom_crating"},
                                    {"label": "专业艺术品处理", "value": "art_handler"},
                                ],
                                inline=True,
                            ),
                        ], width=12),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("预测价格", id="predict-button", color="primary", className="w-100 mt-3"),
                        ], width=12),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("预测结果", className="text-center"),
                dbc.CardBody([
                    html.Div(id="prediction-result", className="text-center"),
                    html.Div(id="confidence-interval", className="text-center mt-2"),
                    dcc.Graph(id="factors-graph"),
                ]),
            ], className="mb-4"),
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("报价列表", className="text-center"),
                dbc.CardBody([
                    html.Div(id="quotes-table"),
                ]),
            ]),
        ], width=12),
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # 每分钟更新一次
        n_intervals=0
    ),
    
], fluid=True)


# 回调函数：加载下拉菜单选项
@callback(
    [Output("origin-dropdown", "options"),
     Output("destination-dropdown", "options"),
     Output("transport-mode-dropdown", "options"),
     Output("cargo-type-dropdown", "options")],
    [Input("interval-component", "n_intervals")]
)
def load_dropdown_options(n):
    # 获取地点数据
    try:
        locations_response = requests.get(f"{API_BASE_URL}/api/v1/quotes/locations")
        locations = locations_response.json()
        location_options = [{"label": loc["name"], "value": loc["id"]} for loc in locations]
    except Exception as e:
        print(f"获取地点数据失败: {e}")
        location_options = []
    
    # 获取运输方式数据
    try:
        transport_modes_response = requests.get(f"{API_BASE_URL}/api/v1/quotes/transport-modes")
        transport_modes = transport_modes_response.json()
        transport_mode_options = [{"label": mode["name"], "value": mode["id"]} for mode in transport_modes]
    except Exception as e:
        print(f"获取运输方式数据失败: {e}")
        transport_mode_options = []
    
    # 获取货物类型数据
    try:
        cargo_types_response = requests.get(f"{API_BASE_URL}/api/v1/quotes/cargo-types")
        cargo_types = cargo_types_response.json()
        cargo_type_options = [{"label": cargo["name"], "value": cargo["id"]} for cargo in cargo_types]
    except Exception as e:
        print(f"获取货物类型数据失败: {e}")
        cargo_type_options = []
    
    return location_options, location_options, transport_mode_options, cargo_type_options


# 回调函数：预测价格
@callback(
    [Output("prediction-result", "children"),
     Output("confidence-interval", "children"),
     Output("factors-graph", "figure")],
    [Input("predict-button", "n_clicks")],
    [State("origin-dropdown", "value"),
     State("destination-dropdown", "value"),
     State("transport-mode-dropdown", "value"),
     State("cargo-type-dropdown", "value"),
     State("weight-input", "value"),
     State("volume-input", "value"),
     State("special-requirements-checklist", "value")]
)
def predict_price(n_clicks, origin_id, destination_id, transport_mode_id, cargo_type_id, weight, volume, special_requirements):
    if n_clicks is None or not all([origin_id, destination_id, transport_mode_id, cargo_type_id, weight, volume]):
        return "请填写所有必填字段", "", go.Figure()
    
    # 准备请求数据
    request_data = {
        "origin_id": origin_id,
        "destination_id": destination_id,
        "transport_mode_id": transport_mode_id,
        "cargo_type_id": cargo_type_id,
        "weight": weight,
        "volume": volume,
        "special_requirements": special_requirements or []
    }
    
    try:
        # 发送预测请求
        response = requests.post(f"{API_BASE_URL}/api/v1/quotes/predict", json=request_data)
        result = response.json()
        
        # 提取预测结果
        predicted_price = result.get("predicted_price", 0)
        confidence_interval = result.get("confidence_interval", [0, 0])
        factors = result.get("factors", {})
        
        # 创建因素图表
        if factors:
            factors_df = pd.DataFrame(list(factors.items()), columns=["Factor", "Importance"])
            factors_df = factors_df.sort_values("Importance", ascending=False)
            fig = px.bar(
                factors_df, 
                x="Importance", 
                y="Factor", 
                orientation="h",
                title="影响因素重要性",
                labels={"Importance": "重要性", "Factor": "因素"},
                color="Importance",
                color_continuous_scale="Viridis",
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="无可用的影响因素数据")
        
        # 格式化输出
        price_text = html.H3(f"预测价格: ¥{predicted_price:.2f}", style={"color": "green"})
        ci_text = f"置信区间: ¥{confidence_interval[0]:.2f} - ¥{confidence_interval[1]:.2f}"
        
        return price_text, ci_text, fig
    
    except Exception as e:
        print(f"预测价格失败: {e}")
        return html.H3("预测失败，请稍后重试", style={"color": "red"}), "", go.Figure()


# 回调函数：加载报价列表
@callback(
    Output("quotes-table", "children"),
    [Input("interval-component", "n_intervals")]
)
def load_quotes(n):
    try:
        # 获取报价列表
        quotes_response = requests.get(f"{API_BASE_URL}/api/v1/quotes/quotes")
        quotes = quotes_response.json()
        
        if not quotes:
            return html.P("暂无报价数据")
        
        # 创建表格
        table_header = [
            html.Thead(html.Tr([
                html.Th("ID"),
                html.Th("客户ID"),
                html.Th("起始地"),
                html.Th("目的地"),
                html.Th("运输方式"),
                html.Th("货物类型"),
                html.Th("重量 (kg)"),
                html.Th("体积 (m³)"),
                html.Th("价格"),
                html.Th("状态"),
                html.Th("创建时间"),
            ]))
        ]
        
        rows = []
        for quote in quotes:
            row = html.Tr([
                html.Td(quote["id"]),
                html.Td(quote["customer_id"]),
                html.Td(quote["origin_id"]),
                html.Td(quote["destination_id"]),
                html.Td(quote["transport_mode_id"]),
                html.Td(quote["cargo_type_id"]),
                html.Td(f"{quote['weight']:.1f}"),
                html.Td(f"{quote['volume']:.1f}"),
                html.Td(f"{quote['price']:.2f} {quote['currency']}"),
                html.Td(quote["status"]),
                html.Td(quote["created_at"].split("T")[0]),
            ])
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)
    
    except Exception as e:
        print(f"加载报价列表失败: {e}")
        return html.P(f"加载报价列表失败: {str(e)}")


# 启动服务器
if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 