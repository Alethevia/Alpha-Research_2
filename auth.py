"""
Streamlit 密码登录门
====================
通过 os.environ.get("APP_PASSWORD") 取参考密码。
未设环境变量时关闭密码保护 (开发友好)。
登录成功后 600 秒内免登录, 之后自动失效需重新输入。

设计要点:
  - 失败显示红色 st.error, 不区分大小写避免泄露细节 → 严格区分大小写以防字典攻击
  - 用 hmac.compare_digest 防止时序侧信道泄露
  - 状态全部存 st.session_state, 无外部存储
  - require_auth() 在任何敏感渲染前调用; 未通过则 st.stop() 阻断后续脚本
"""

from __future__ import annotations

import hmac
import os
import time

import streamlit as st


SESSION_TTL_SECONDS = 600
_AUTH_TS_KEY = "_auth_ts"
_AUTH_OK_KEY = "_auth_ok"


def _reference_password() -> str | None:
    pw = os.environ.get("APP_PASSWORD", "")
    pw = pw.strip()
    return pw if pw else None


def is_auth_enabled() -> bool:
    return _reference_password() is not None


def _session_valid() -> bool:
    if not st.session_state.get(_AUTH_OK_KEY):
        return False
    ts = st.session_state.get(_AUTH_TS_KEY, 0)
    return (time.time() - ts) < SESSION_TTL_SECONDS


def _do_login(attempt: str) -> bool:
    ref = _reference_password()
    if ref is None:
        return True
    ok = hmac.compare_digest(attempt or "", ref)
    if ok:
        st.session_state[_AUTH_OK_KEY] = True
        st.session_state[_AUTH_TS_KEY] = time.time()
    return ok


def require_auth() -> None:
    """
    必须在 app 顶部调用 (在任何 st.title / 数据加载之前)。
    未启用密码 → 立即返回, 不影响开发。
    已登录且未过期 → 显示左上角剩余时间徽标后返回。
    否则渲染登录页 + st.stop() 阻断脚本。
    """
    if not is_auth_enabled():
        return

    if _session_valid():
        remaining = SESSION_TTL_SECONDS - int(time.time() - st.session_state[_AUTH_TS_KEY])
        st.sidebar.caption(f"🔐 会话剩余: {remaining} 秒 (Session expires in {remaining}s)")
        if st.sidebar.button("登出 / Logout", key="_auth_logout_btn"):
            st.session_state.pop(_AUTH_OK_KEY, None)
            st.session_state.pop(_AUTH_TS_KEY, None)
            st.rerun()
        return

    # ===== Render login form, then BLOCK =====
    st.markdown("# 🔐 访问受限 / Restricted Access")
    st.markdown(
        "本工具仅供授权用户使用。请输入访问密码。\n\n"
        "_This tool requires authorization. Enter access password._"
    )
    st.caption(f"会话有效期: {SESSION_TTL_SECONDS} 秒 / Session lasts {SESSION_TTL_SECONDS}s after login.")

    with st.form("_auth_form", clear_on_submit=False):
        pw = st.text_input("密码 / Password", type="password", key="_auth_pw_input")
        submitted = st.form_submit_button("登录 / Login", type="primary", use_container_width=True)

    if submitted:
        if _do_login(pw):
            st.success("✅ 登录成功 / Login successful — 正在加载...")
            time.sleep(0.4)
            st.rerun()
        else:
            st.error("❌ 密码错误 / Wrong password")

    # 阻断后续脚本执行
    st.stop()
