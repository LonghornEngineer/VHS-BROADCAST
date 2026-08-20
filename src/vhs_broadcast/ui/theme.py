"""Retro broadcast-equipment stylesheet."""
from __future__ import annotations

from .. import config as C

STYLESHEET = f"""
* {{
    font-family: "Consolas", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 12px;
    color: {C.TEXT};
}}
QMainWindow, QWidget#Root {{
    background-color: {C.BG};
}}
QLabel {{ color: {C.TEXT}; }}
QLabel#Title {{
    color: {C.GREEN};
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 3px;
}}
QLabel#Value {{ color: {C.AMBER}; font-weight: bold; }}
QLabel#SectionHint {{ color: {C.TEXT_DIM}; letter-spacing: 2px; }}

QGroupBox {{
    background-color: {C.PANEL};
    border: 1px solid {C.BORDER};
    border-radius: 3px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 1px 8px;
    color: {C.GREEN};
    background-color: {C.PANEL_HI};
    border: 1px solid {C.BORDER};
    letter-spacing: 2px;
    font-weight: bold;
}}

QPushButton {{
    background-color: {C.PANEL_HI};
    border: 1px solid {C.BORDER};
    border-radius: 2px;
    padding: 6px 12px;
    color: {C.TEXT};
    letter-spacing: 1px;
}}
QPushButton:hover {{ border-color: {C.GREEN}; color: {C.GREEN}; }}
QPushButton:pressed {{ background-color: {C.GREEN_DIM}; }}
QPushButton:checked {{
    background-color: {C.GREEN_DIM};
    border-color: {C.GREEN};
    color: #001b06;
    font-weight: bold;
}}
QPushButton#Rec:checked {{
    background-color: {C.RED};
    border-color: {C.RED};
    color: #1a0000;
}}

QSlider::groove:horizontal {{
    height: 4px; background: #0a1a0c; border: 1px solid {C.BORDER};
}}
QSlider::sub-page:horizontal {{ background: {C.GREEN_DIM}; }}
QSlider::handle:horizontal {{
    background: {C.GREEN}; border: 1px solid #0a1a0c;
    width: 10px; margin: -6px 0; border-radius: 1px;
}}
QSlider::handle:horizontal:hover {{ background: {C.CYAN}; }}

QComboBox {{
    background: {C.PANEL_HI}; border: 1px solid {C.BORDER};
    padding: 2px 6px; color: {C.TEXT};
}}
QComboBox QAbstractItemView {{
    background: {C.PANEL}; border: 1px solid {C.BORDER};
    selection-background-color: {C.GREEN_DIM};
}}
QLineEdit {{
    background: #061006; border: 1px solid {C.BORDER};
    padding: 4px; color: {C.AMBER};
}}
QCheckBox {{ color: {C.TEXT}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {C.BORDER}; background: #061006;
}}
QCheckBox::indicator:checked {{ background: {C.GREEN}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {C.PANEL}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C.BORDER}; min-height: 20px; }}
QToolTip {{ background: {C.PANEL}; color: {C.GREEN}; border: 1px solid {C.BORDER}; }}

/* --- fixes: transparent scroll area (no white bg) + compact controls --- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QComboBox {{
    background-color: {C.PANEL_HI}; color: {C.GREEN};
    border: 1px solid {C.BORDER}; padding: 1px 6px; min-height: 16px;
}}
QComboBox QAbstractItemView {{
    background-color: {C.PANEL}; color: {C.GREEN};
    selection-background-color: {C.GREEN_DIM};
}}
QGroupBox {{ margin-top: 8px; padding: 4px 6px 3px 6px; }}
QSlider {{ min-height: 11px; max-height: 13px; }}
QSlider::groove:horizontal {{ height: 3px; }}
QSlider::handle:horizontal {{ width: 9px; height: 9px; margin: -4px 0; }}
QCheckBox {{ spacing: 4px; }}

/* effect name lives on the enable checkbox: bold, bright when on, dim when off */
QCheckBox#EffectTitle {{ font-weight: bold; letter-spacing: 1px; color: {C.TEXT_DIM}; }}
QCheckBox#EffectTitle:checked {{ color: {C.GREEN}; }}

/* --- dropdown / popup menus (INPUT picker) -------------------------------- */
QMenu {{
    background-color: {C.PANEL};
    color: {C.TEXT};
    border: 1px solid {C.BORDER};
    padding: 3px;
}}
QMenu::item {{
    padding: 5px 26px 5px 14px;
    background: transparent;
}}
QMenu::item:selected {{
    background-color: {C.GREEN_DIM};
    color: #001b06;
}}
QMenu::item:disabled {{ color: {C.TEXT_DIM}; }}
QMenu::separator {{ height: 1px; background: {C.BORDER}; margin: 4px 8px; }}
QMenu::right-arrow {{ width: 10px; height: 10px; }}
QMenu::indicator {{ width: 14px; height: 14px; }}

QToolButton {{
    background-color: {C.PANEL_HI}; color: {C.GREEN};
    border: 1px solid {C.BORDER}; padding: 5px 9px;
}}
QToolButton:hover {{ background-color: {C.GREEN_DIM}; color: #001b06; }}
QToolButton::menu-indicator {{ image: none; width: 0px; }}

/* modulation buttons light up when their modulation is enabled */
QPushButton#ModBtn, QPushButton#AudioBtn {{
    padding: 0px; font-weight: bold;
}}
QPushButton#ModBtn[active="true"] {{
    background-color: {C.GREEN_DIM}; color: #001b06; border-color: {C.GREEN};
}}
QPushButton#AudioBtn[active="true"] {{
    background-color: {C.AMBER}; color: #1a0f00; border-color: {C.AMBER};
}}
/* popup modulation editors */
QWidget#ModPanel {{ background-color: {C.PANEL}; min-width: 300px; }}
QLabel#ModHead {{
    color: {C.GREEN}; font-weight: bold; letter-spacing: 1px;
    padding-bottom: 2px;
}}

/* dialogs / message boxes (were OS-default light -> unreadable green text) */
QMessageBox, QDialog {{ background-color: {C.BG}; color: {C.TEXT}; }}
QMessageBox QLabel, QDialog QLabel {{ color: {C.TEXT}; background: transparent; }}
QMessageBox QPushButton, QDialog QPushButton {{
    background-color: {C.PANEL_HI}; color: {C.GREEN};
    border: 1px solid {C.BORDER}; padding: 5px 16px; min-width: 64px;
}}
QMessageBox QPushButton:hover, QDialog QPushButton:hover {{
    background-color: {C.GREEN_DIM}; color: #001b06;
}}
QMessageBox QPushButton:default, QDialog QPushButton:default {{
    border: 1px solid {C.GREEN};
}}

/* progress bar (render dialog) + tooltips -> match the dark theme */
QProgressBar {{
    background-color: #0a1a0c; border: 1px solid {C.BORDER};
    color: {C.TEXT}; text-align: center; height: 16px;
}}
QProgressBar::chunk {{ background-color: {C.GREEN_DIM}; }}
QToolTip {{
    background-color: {C.PANEL}; color: {C.TEXT};
    border: 1px solid {C.BORDER}; padding: 3px 6px;
}}

QLabel#Stats {{ color: {C.TEXT_DIM}; font-family: monospace; padding: 2px 2px; }}

/* tabs + signal-chain list */
QTabWidget::pane {{ border: 1px solid {C.BORDER}; top: -1px; }}
QTabBar::tab {{
    background: {C.PANEL}; color: {C.TEXT_DIM};
    border: 1px solid {C.BORDER}; padding: 5px 14px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {C.PANEL_HI}; color: {C.GREEN}; }}
QListWidget#ChainList {{
    background: {C.PANEL}; color: {C.TEXT};
    border: 1px solid {C.BORDER}; font-family: monospace; outline: 0;
}}
QListWidget#ChainList::item {{ padding: 5px 4px; border-bottom: 1px solid #14241a; }}
QListWidget#ChainList::item:selected {{ background: {C.GREEN_DIM}; color: #001b06; }}

/* signal-chain tab + reorder list */
QTabWidget::pane {{ border: 1px solid {C.BORDER}; top: -1px; }}
QTabBar::tab {{
    background: {C.PANEL}; color: {C.TEXT_DIM};
    border: 1px solid {C.BORDER}; padding: 5px 14px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {C.PANEL_HI}; color: {C.GREEN}; }}
QListWidget#ChainList {{
    background: {C.BG}; color: {C.TEXT};
    border: 1px solid {C.BORDER}; outline: none;
    font-family: monospace;
}}
QListWidget#ChainList::item {{ padding: 4px 6px; }}
QListWidget#ChainList::item:selected {{ background: {C.PANEL_HI}; color: {C.GREEN}; }}

/* larger playback scrub bar */
QSlider#Scrub::groove:horizontal {{ height: 8px; background: #0a1a0c; border: 1px solid {C.BORDER}; border-radius: 3px; }}
QSlider#Scrub::sub-page:horizontal {{ background: {C.GREEN_DIM}; border: 1px solid {C.BORDER}; border-radius: 3px; }}
QSlider#Scrub::handle:horizontal {{ background: {C.GREEN}; width: 12px; margin: -5px 0; border-radius: 3px; }}

QSpinBox {{ background: {C.BG}; color: {C.TEXT}; border: 1px solid {C.BORDER}; padding: 2px 4px; }}
QSpinBox::up-button, QSpinBox::down-button {{ background: {C.PANEL_HI}; width: 14px; border: 1px solid {C.BORDER}; }}
QSpinBox::up-arrow {{ image: none; width:0;height:0; border-left:3px solid transparent; border-right:3px solid transparent; border-bottom:5px solid {C.GREEN}; }}
QSpinBox::down-arrow {{ image: none; width:0;height:0; border-left:3px solid transparent; border-right:3px solid transparent; border-top:5px solid {C.GREEN}; }}
"""
