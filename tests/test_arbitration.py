"""多来源读数仲裁：source_id 上报留存、来源优先级版本（expected_version /
生效时刻）、权威时间线、并列冲突不进入剂量与诊断、归并合并来源配置。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.receipts import content_hash_of

BASE = 1_700_000_000  # 2023-11-14T22:13:20Z


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def make_exhibit(client: TestClient, limit: float = 1000.0, interval: float = 3600.0,
                 eid: str | None = None, name: str = "唐代绢画") -> str:
    body = {"name": name, "dose_limit_luxh": limit, "max_interval_seconds": interval}
    if eid is not None:
        body["id"] = eid
    resp = client.post("/exhibits", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def send(client: TestClient, exhibit_id: str, readings: list[dict]) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def put_priority(client: TestClient, exhibit_id: str, priorities: list[str],
                 effective_at: float, expected_version: int) -> dict:
    resp = client.put(f"/exhibits/{exhibit_id}/source-priority", json={
        "effective_at": iso(effective_at),
        "priorities": priorities,
        "expected_version": expected_version,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def dose(client: TestClient, exhibit_id: str, start: float, end: float,
         extra: dict | None = None) -> dict:
    params = {"start": iso(start), "end": iso(end)}
    if extra:
        params.update(extra)
    resp = client.get(f"/exhibits/{exhibit_id}/dose", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def timeline(client: TestClient, exhibit_id: str, start: float, end: float) -> dict:
    resp = client.get(f"/exhibits/{exhibit_id}/timeline",
                      params={"start": iso(start), "end": iso(end)})
    assert resp.status_code == 200, resp.text
    return resp.json()


def diagnose(client: TestClient, exhibit_id: str, start: float, end: float,
             max_rate: float = 1000.0, stuck_tol: float = 0.0,
             min_stuck: float = 10_000_000.0) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings/diagnose", json={
        "start": iso(start), "end": iso(end),
        "max_rate_lux_per_second": max_rate,
        "stuck_tolerance_lux": stuck_tol,
        "min_stuck_seconds": min_stuck,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def merge_commit(client: TestClient, source_id: str, target_id: str,
                 resolutions: list[dict] | None = None) -> dict:
    preview = client.post("/merges/preview",
                          json={"source_id": source_id, "target_id": target_id})
    assert preview.status_code == 200, preview.text
    resp = client.post("/merges/commit", json={
        "token": preview.json()["token"], "resolutions": resolutions or [],
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------- 上报：source_id 留存与旧行为兼容 ----------

def test_ingest_sources_kept_and_listed(client):
    eid = make_exhibit(client)
    out = send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "sensor-a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "sensor-b"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "sensor-a"},
    ])
    assert out["inserted"] == 3
    # 同一时刻不同来源全部留存
    resp = client.get(f"/exhibits/{eid}/readings")
    body = resp.json()["readings"]
    assert [(r["id"], r["source_id"]) for r in body] == [
        ("r1", "sensor-a"), ("r2", "sensor-b"), ("r3", "sensor-a"),
    ]


def test_legacy_ingest_without_source_unchanged(client):
    eid = make_exhibit(client)
    out = send(client, eid, [{"id": "r1", "ts": iso(BASE), "lux": 100}])
    # 旧式请求的收据内容不含 source_id 键，内容哈希可重算复现
    content = out["receipt"]["content"]
    assert "source_id" not in content["request"]["readings"][0]
    assert out["receipt"]["content_hash"] == content_hash_of(content)
    # 幂等语义不变：同 ID 同数据 duplicate、不同数据 conflict
    out2 = send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r1", "ts": iso(BASE), "lux": 101},
    ])
    assert [r["status"] for r in out2["results"]] == ["duplicate", "conflict"]
    # 读数列表带 source_id 字段（未声明来源为 null）
    rows = client.get(f"/exhibits/{eid}/readings").json()["readings"]
    assert rows[0]["source_id"] is None


def test_same_id_different_source_is_conflict(client):
    eid = make_exhibit(client)
    send(client, eid, [{"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"}])
    out = send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "b"},   # 来源不同
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},   # 完全一致
        {"id": "r1", "ts": iso(BASE), "lux": 100},                     # 缺来源
    ])
    assert [r["status"] for r in out["results"]] == ["conflict", "duplicate", "conflict"]


def test_receipt_records_source_id_and_hash_recomputable(client):
    eid = make_exhibit(client)
    out = send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "sensor-a"},
    ])
    content = out["receipt"]["content"]
    assert content["request"]["readings"][0]["source_id"] == "sensor-a"
    assert out["receipt"]["content_hash"] == content_hash_of(content)
    verify = client.get(f"/exhibits/{eid}/receipts/verify").json()
    assert verify["ok"] is True


# ---------- 来源优先级版本：发布、expected_version、校验 ----------

def test_put_and_get_priority_versions(client):
    eid = make_exhibit(client)
    v1 = put_priority(client, eid, ["sensor-a", "sensor-b"], BASE, 0)
    assert v1["version"] == 1
    assert v1["priorities"] == ["sensor-a", "sensor-b"]
    assert v1["canonical_id"] == eid

    got = client.get(f"/exhibits/{eid}/source-priority")
    assert got.status_code == 200
    body = got.json()
    assert body["current_version"] == 1
    assert [v["version"] for v in body["versions"]] == [1]

    # 过期 expected_version → 409
    stale = client.put(f"/exhibits/{eid}/source-priority", json={
        "effective_at": iso(BASE + 100), "priorities": ["sensor-b"],
        "expected_version": 0,
    })
    assert stale.status_code == 409

    v2 = put_priority(client, eid, ["sensor-b", "sensor-a"], BASE + 3600, 1)
    assert v2["version"] == 2
    body = client.get(f"/exhibits/{eid}/source-priority").json()
    assert body["current_version"] == 2
    assert [v["version"] for v in body["versions"]] == [1, 2]


def test_get_priority_empty_when_unconfigured(client):
    eid = make_exhibit(client)
    body = client.get(f"/exhibits/{eid}/source-priority").json()
    assert body["current_version"] == 0
    assert body["versions"] == []


def test_put_priority_validation_and_404(client):
    eid = make_exhibit(client)
    # 优先级列表来源重复 → 422
    resp = client.put(f"/exhibits/{eid}/source-priority", json={
        "effective_at": iso(BASE), "priorities": ["a", "a"], "expected_version": 0,
    })
    assert resp.status_code == 422
    # 缺 expected_version → 422
    resp = client.put(f"/exhibits/{eid}/source-priority", json={
        "effective_at": iso(BASE), "priorities": ["a"],
    })
    assert resp.status_code == 422
    # 展品不存在 → 404
    assert client.put("/exhibits/nope/source-priority", json={
        "effective_at": iso(BASE), "priorities": ["a"], "expected_version": 0,
    }).status_code == 404
    assert client.get("/exhibits/nope/source-priority").status_code == 404
    assert client.get("/exhibits/nope/timeline",
                      params={"start": iso(BASE), "end": iso(BASE + 1)}).status_code == 404


def test_concurrent_put_priority_only_one_wins(client):
    eid = make_exhibit(client)

    def put():
        return client.put(f"/exhibits/{eid}/source-priority", json={
            "effective_at": iso(BASE), "priorities": ["a"], "expected_version": 0,
        }).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(lambda _: put(), range(2)))
    assert sorted(codes) == [201, 409]
    assert client.get(f"/exhibits/{eid}/source-priority").json()["current_version"] == 1


# ---------- 仲裁：权威值、并列冲突、生效时刻 ----------

def test_dose_arbitrated_by_highest_priority(client):
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "b"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    got = dose(client, eid, BASE, BASE + 3600)
    # 权威值取高优先级来源 a：100 lux 恒定 1 小时 = 100 lux·h
    assert got["known_dose_luxh"] == 100.0
    assert got["covered_seconds"] == 3600.0
    assert got["reading_count"] == 2  # 权威点数量，而非原始读数条数

    tl = timeline(client, eid, BASE, BASE + 3600)
    assert tl["arbitrated"] is True
    assert [(e["status"], e.get("lux")) for e in tl["entries"]] == [
        ("ok", 100.0), ("ok", 100.0),
    ]
    first = tl["entries"][0]
    assert first["reading_ids"] == ["r1"]          # 只有胜出集合进入时间线
    assert first["source_ids"] == ["a"]
    assert first["priority_version"] == 1


def test_tied_unlisted_sources_conflict_excluded(client):
    """未列入优先级的来源并列最低：照度不同时该时刻保持冲突。

    冲突状态保留在权威时间线中供查询，但冲突读数既不进入剂量积分
    （两侧权威点不直接相连），也不进入诊断：不生成 timestamp_conflict，
    诊断连续链在两侧重置（不形成跨冲突的 gap / jump / stuck）。
    """
    eid = make_exhibit(client, interval=1800.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE + 1800), "lux": 100, "source_id": "c"},
        {"id": "r3", "ts": iso(BASE + 1800), "lux": 200, "source_id": "d"},
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a"], BASE - 10, 0)

    got = dose(client, eid, BASE, BASE + 3600)
    # 冲突时刻把 BASE 与 BASE+3600 切成两段：间隔 3600 > 1800 且跨冲突，
    # 两侧权威点绝不直接相连 → 整段缺测
    assert got["covered_seconds"] == 0.0
    assert got["unknown_seconds"] == 3600.0
    assert got["known_dose_luxh"] == 0.0
    assert got["reading_count"] == 2

    # 诊断：仲裁冲突不是 timestamp_conflict，也不形成跨冲突的 gap；
    # 连续链在两侧重置（每侧只剩一个点，无任何事件）
    events = diagnose(client, eid, BASE, BASE + 3600)["events"]
    assert events == []

    tl = timeline(client, eid, BASE, BASE + 3600)
    statuses = [e["status"] for e in tl["entries"]]
    assert statuses == ["ok", "conflict", "ok"]
    mid = tl["entries"][1]
    assert mid["lux_span"] == 100.0
    assert "lux" not in mid  # 冲突时刻无权威值


def test_tied_same_lux_merged(client):
    """并列最高但照度相同：合并为一个权威点（读数 ID 全部保留）。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "c"},
        {"id": "r2", "ts": iso(BASE), "lux": 100, "source_id": "d"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a"], BASE - 10, 0)
    got = dose(client, eid, BASE, BASE + 3600)
    assert got["known_dose_luxh"] == 100.0
    assert got["covered_seconds"] == 3600.0
    entry = timeline(client, eid, BASE, BASE + 3600)["entries"][0]
    assert entry["status"] == "ok"
    assert entry["reading_ids"] == ["r1", "r2"]
    assert entry["source_ids"] == ["c", "d"]


def test_same_source_same_ts_different_lux_conflicts(client):
    """同一来源在同一时刻上报不同照度（不同读数 ID）：同样保持冲突。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 200, "source_id": "a"},
    ])
    put_priority(client, eid, ["a"], BASE - 10, 0)
    entry = timeline(client, eid, BASE, BASE + 1)["entries"][0]
    assert entry["status"] == "conflict"
    assert entry["reading_ids"] == ["r1", "r2"]


def test_unlisted_and_null_source_are_lowest(client):
    """未列入优先级的来源与未声明来源的旧式读数并列最低。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "unlisted"},
        {"id": "r3", "ts": iso(BASE), "lux": 900},  # 未声明来源
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a"], BASE - 10, 0)
    assert dose(client, eid, BASE, BASE + 3600)["known_dose_luxh"] == 100.0
    entry = timeline(client, eid, BASE, BASE + 1)["entries"][0]
    assert entry["reading_ids"] == ["r1"]


def test_effective_at_selects_version_per_instant(client):
    """每个采样时刻按 effective_at <= ts 的最新版本仲裁；生效前平权。"""
    eid = make_exhibit(client, interval=7200.0)
    put_priority(client, eid, ["a", "b"], BASE, 0)
    put_priority(client, eid, ["b", "a"], BASE + 3600, 1)
    send(client, eid, [
        # 首个版本生效之前：平权，同值合并
        {"id": "r1", "ts": iso(BASE - 100), "lux": 50, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE - 100), "lux": 50, "source_id": "b"},
        # v1 生效：a 优先
        {"id": "r3", "ts": iso(BASE + 1800), "lux": 100, "source_id": "a"},
        {"id": "r4", "ts": iso(BASE + 1800), "lux": 200, "source_id": "b"},
        # v2 生效：b 优先
        {"id": "r5", "ts": iso(BASE + 7200), "lux": 100, "source_id": "a"},
        {"id": "r6", "ts": iso(BASE + 7200), "lux": 200, "source_id": "b"},
    ])
    entries = timeline(client, eid, BASE - 100, BASE + 7200)["entries"]
    assert [(e["status"], e.get("lux"), e["priority_version"]) for e in entries] == [
        ("ok", 50.0, None),
        ("ok", 100.0, 1),
        ("ok", 200.0, 2),
    ]


def test_priority_adjustment_recomputes_from_effective_at_only(client):
    """调整优先级只从生效时刻起影响重算；历史收据与原始读数不改写。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "b"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 500, "source_id": "b"},
        {"id": "r5", "ts": iso(BASE + 7200), "lux": 100, "source_id": "a"},
        {"id": "r6", "ts": iso(BASE + 7200), "lux": 500, "source_id": "b"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    before = dose(client, eid, BASE, BASE + 7200)
    assert before["known_dose_luxh"] == 200.0  # a 全程优先：100 lux × 2h

    # 自 BASE+3600 起 b 优先：只影响生效时刻之后的重算
    put_priority(client, eid, ["b", "a"], BASE + 3600, 1)
    after = dose(client, eid, BASE, BASE + 7200)
    # (100+500)/2×1h + 500×1h = 800
    assert after["known_dose_luxh"] == 800.0

    # 历史收据链不改写，整链校验仍通过
    verify = client.get(f"/exhibits/{eid}/receipts/verify").json()
    assert verify["ok"] is True
    # 原始读数不改写：b 的低优先级读数原样保留
    rows = {r["id"]: r for r in client.get(f"/exhibits/{eid}/readings").json()["readings"]}
    assert rows["r2"]["lux"] == 500.0 and rows["r2"]["source_id"] == "b"


def test_empty_priorities_means_flat(client):
    """空优先级列表：全部来源平权（同值合并、异值冲突）。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 200, "source_id": "b"},
    ])
    put_priority(client, eid, [], BASE - 10, 0)
    tl = timeline(client, eid, BASE, BASE + 1)
    assert tl["arbitrated"] is True
    assert tl["entries"][0]["status"] == "conflict"


# ---------- 权威时间线查询 ----------

def test_timeline_flat_without_config(client):
    """未配置任何版本：平权口径，与旧行为一致（同值合并、异值冲突）。"""
    eid = make_exhibit(client)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE), "lux": 200},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 150},
    ])
    tl = timeline(client, eid, BASE, BASE + 3600)
    assert tl["arbitrated"] is False
    assert [e["status"] for e in tl["entries"]] == ["conflict", "ok"]
    assert tl["entries"][0]["priority_version"] is None
    assert tl["entries"][0]["lux_span"] == 100.0
    assert tl["entries"][1]["lux"] == 150.0


def test_timeline_window_and_validation(client):
    eid = make_exhibit(client)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 3600), "lux": 100},
    ])
    tl = timeline(client, eid, BASE, BASE + 1)
    assert [e["ts"] for e in tl["entries"]] == [iso(BASE).replace("+00:00", "Z")]
    # end <= start → 422
    resp = client.get(f"/exhibits/{eid}/timeline",
                      params={"start": iso(BASE + 1), "end": iso(BASE)})
    assert resp.status_code == 422


def test_dose_after_point_skips_conflict(client):
    """窗口终点之后最近权威点：跳过保持冲突的时刻，且跨冲突对不积分。

    窗口 [BASE, BASE+1900] 之后 BASE+2700 是仲裁冲突时刻，再往后
    BASE+3600 才有权威点：窗口末端的裁剪锚点仍是 BASE+3600 的权威点
    （冲突时刻被 first_point_after 跳过），但 (BASE, BASE+3600) 这一对
    跨过冲突断点，梯形积分在断点处断开——窗口不与任何有效积分段相交，
    全部计为缺测，绝不把两侧读数直接相连。
    """
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE + 2700), "lux": 100, "source_id": "c"},
        {"id": "r3", "ts": iso(BASE + 2700), "lux": 200, "source_id": "d"},
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a"], BASE - 10, 0)
    got = dose(client, eid, BASE, BASE + 1900)
    assert got["covered_seconds"] == 0.0
    assert got["unknown_seconds"] == 1900.0
    assert got["known_dose_luxh"] == 0.0

    # 冲突时刻仍保留在权威时间线中，只是不作为积分 / 诊断输入
    tl = timeline(client, eid, BASE, BASE + 3600)
    assert [e["status"] for e in tl["entries"]] == ["ok", "conflict", "ok"]


# ---------- 仲裁冲突传播：剂量积分必须在冲突处断开（回归） ----------

def test_middle_conflict_severs_dose_between_authoritative_points(client):
    """同一采样时刻最高优先级来源照度不同且位于两个权威点之间：
    即使两侧权威点间隔未超过最大采样间隔，积分也必须在冲突处断开，
    绝不把两侧读数直接相连。"""
    eid = make_exhibit(client, interval=7200.0)  # 3600s 间隔远未超限
    send(client, eid, [
        {"id": "a0", "ts": iso(BASE - 3600), "lux": 100, "source_id": "a"},
        {"id": "a1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "c2", "ts": iso(BASE + 3600), "lux": 100, "source_id": "c"},
        {"id": "d2", "ts": iso(BASE + 3600), "lux": 900, "source_id": "d"},
        {"id": "a3", "ts": iso(BASE + 7200), "lux": 200, "source_id": "a"},
        {"id": "a4", "ts": iso(BASE + 10800), "lux": 200, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10_000, 0)

    # 权威时间线仍保留冲突状态
    tl = timeline(client, eid, BASE - 3600, BASE + 10800)
    assert [(e["status"], e.get("lux")) for e in tl["entries"]] == [
        ("ok", 100.0), ("ok", 100.0), ("conflict", None),
        ("ok", 200.0), ("ok", 200.0),
    ]

    # 旧的（错误）行为：移除冲突点后 (100@0 → 200@2h) 直接相连，
    # 跨冲突窗口会被当作有效梯形段。正确行为：冲突时刻无权威值，
    # 任何触及它的相邻对都不是积分段——窗口覆盖按断点两侧的各段统计。
    got = dose(client, eid, BASE, BASE + 7200)
    assert got["known_dose_luxh"] == 0.0          # (0,冲突) 与 (冲突,2h) 都无端点
    assert got["covered_seconds"] == 0.0
    assert got["unknown_seconds"] == 7200.0
    assert got["reading_count"] == 2              # 只计窗口内权威点

    # 冲突前一段、冲突后一段各自独立成立
    left = dose(client, eid, BASE - 3600, BASE)
    assert left["known_dose_luxh"] == 100.0
    assert left["covered_seconds"] == 3600.0
    right = dose(client, eid, BASE + 7200, BASE + 10800)
    assert right["known_dose_luxh"] == 200.0
    assert right["covered_seconds"] == 3600.0

    # 窗口完全落在冲突左侧 / 右侧的有效段内：仍可线性裁剪
    pre = dose(client, eid, BASE - 1800, BASE - 900)
    assert pre["known_dose_luxh"] == 25.0         # 100 lux × 0.25h
    assert pre["covered_seconds"] == 900.0
    post = dose(client, eid, BASE + 9000, BASE + 9900)
    assert post["known_dose_luxh"] == 50.0        # 200 lux × 0.25h
    assert post["covered_seconds"] == 900.0


def test_conflict_at_window_boundaries(client):
    """冲突时刻落在窗口边界（起点 / 终点）时积分同样断开。"""
    eid = make_exhibit(client, interval=7200.0)
    send(client, eid, [
        {"id": "am", "ts": iso(BASE - 7200), "lux": 100, "source_id": "a"},
        {"id": "a0", "ts": iso(BASE - 3600), "lux": 100, "source_id": "a"},
        {"id": "c1", "ts": iso(BASE), "lux": 100, "source_id": "c"},
        {"id": "d1", "ts": iso(BASE), "lux": 900, "source_id": "d"},
        {"id": "a2", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
        {"id": "c3", "ts": iso(BASE + 7200), "lux": 100, "source_id": "c"},
        {"id": "d3", "ts": iso(BASE + 7200), "lux": 900, "source_id": "d"},
        {"id": "a4", "ts": iso(BASE + 10800), "lux": 100, "source_id": "a"},
        {"id": "a5", "ts": iso(BASE + 14400), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10_000, 0)
    # 有效权威段：[-2h,-1h]（100）、[3h,4h]（100）；冲突在 0 与 2h。

    # 起点即冲突时刻：相邻对 (冲突→1h) 无起点端点，不积分
    at_start = dose(client, eid, BASE, BASE + 3600)
    assert at_start["known_dose_luxh"] == 0.0
    assert at_start["covered_seconds"] == 0.0
    assert at_start["unknown_seconds"] == 3600.0

    # 终点即冲突时刻：相邻对 (-1h→冲突) 无终点端点，不积分
    at_end = dose(client, eid, BASE - 3600, BASE)
    assert at_end["known_dose_luxh"] == 0.0
    assert at_end["covered_seconds"] == 0.0
    assert at_end["unknown_seconds"] == 3600.0

    # 但冲突左侧 / 右侧的完整有效段照常积分（不向冲突方向延伸）
    left_run = dose(client, eid, BASE - 7200, BASE - 3600)
    assert left_run["known_dose_luxh"] == 100.0
    assert left_run["covered_seconds"] == 3600.0
    right_run = dose(client, eid, BASE + 10800, BASE + 14400)
    assert right_run["known_dose_luxh"] == 100.0
    assert right_run["covered_seconds"] == 3600.0

    # 窗口从冲突时刻起、延伸进右侧完整段：只有段内部分可覆盖，
    # 冲突到段首的 1h 计缺测
    into_run = dose(client, eid, BASE + 7200, BASE + 14400)
    assert into_run["known_dose_luxh"] == 100.0
    assert into_run["covered_seconds"] == 3600.0
    assert into_run["unknown_seconds"] == 3600.0
    assert into_run["reading_count"] == 2

    # 夹在两个冲突之间的孤立点（1h）：两侧都无端点 → 全缺测
    inner = dose(client, eid, BASE + 1800, BASE + 5400)
    assert inner["known_dose_luxh"] == 0.0
    assert inner["covered_seconds"] == 0.0
    assert inner["unknown_seconds"] == 3600.0


def test_conflict_severs_history_cumulative_and_first_exceeded(client):
    """截至窗口末的累计剂量与首次越线按断点后的各段分别累计（顺延不重置，
    也不跨冲突相连）。"""
    eid = make_exhibit(client, limit=300.0, interval=7200.0)
    send(client, eid, [
        {"id": "a0", "ts": iso(BASE - 3600), "lux": 100, "source_id": "a"},
        {"id": "a1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "c2", "ts": iso(BASE + 3600), "lux": 100, "source_id": "c"},
        {"id": "d2", "ts": iso(BASE + 3600), "lux": 900, "source_id": "d"},
        {"id": "a3", "ts": iso(BASE + 7200), "lux": 400, "source_id": "a"},
        {"id": "a4", "ts": iso(BASE + 10800), "lux": 400, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10_000, 0)

    # 截至窗口末：有效历史 = 冲突前 100×1h + 冲突后 400×1h = 500 lux·h；
    # 旧的（错误）直连行为会把 (100@0 → 400@2h) 相连成 500 lux·h 再加
    # 100×1h + 400×1h，累计 1000 lux·h，首次越线时刻完全不同。
    # 窗口 [BASE-3600, BASE+10800] 覆盖：冲突前 100×1h + 冲突后 400×1h = 500
    got = dose(client, eid, BASE - 3600, BASE + 10800)
    assert got["known_dose_luxh"] == 500.0
    assert got["covered_seconds"] == 7200.0
    assert got["unknown_seconds"] == 7200.0       # 4h 窗口：覆盖 2h、跨冲突 2h
    assert got["remaining_luxh"] == -200.0
    assert got["exceeded"] is True
    # 历史累计在 (BASE+7200, BASE+10800) 的 400 lux 恒定段达到 500：
    # 前 100 lux·h 来自冲突前，再 200 lux·h 越线 → 段内 0.5h
    first_at = datetime.fromisoformat(got["first_exceeded_at"]).timestamp()
    assert abs(first_at - (BASE + 7200 + 1800)) < 1.0
    # 首次越线时刻不随窗口缩窄改变：缩窄窗口但仍覆盖越线时刻，值不变
    narrow = dose(client, eid, BASE + 9000, BASE + 10800)
    assert narrow["first_exceeded_at"] == got["first_exceeded_at"]
    # 窗口末早于越线时刻：按口径返回 null（与既有全历史语义一致）
    before_exceed = dose(client, eid, BASE + 3600, BASE + 7200)
    assert before_exceed["first_exceeded_at"] is None
    assert before_exceed["known_dose_luxh"] == 0.0
    assert before_exceed["covered_seconds"] == 0.0


def test_calendar_display_open_periods_broken_at_conflict(client):
    """calendar=display：开放时段的梯形裁剪在仲裁冲突处断开，
    covered / unknown / closed 按断点后的各段分别统计。"""
    # 以 UTC 整点对齐的“当日”为基准（例外日整体替换周规则，避免周几 / 分钟偏移）
    day0 = datetime(2023, 11, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    eid = make_exhibit(client, interval=7200.0)
    send(client, eid, [
        # 01:00 权威点（开放段内）；02:00 仲裁冲突；03:00、04:00、05:00 权威点
        {"id": "a1", "ts": iso(day0 + 3600), "lux": 100, "source_id": "a"},
        {"id": "c2", "ts": iso(day0 + 7200), "lux": 100, "source_id": "c"},
        {"id": "d2", "ts": iso(day0 + 7200), "lux": 900, "source_id": "d"},
        {"id": "a3", "ts": iso(day0 + 10800), "lux": 100, "source_id": "a"},
        {"id": "a4", "ts": iso(day0 + 14400), "lux": 100, "source_id": "a"},
        {"id": "a5", "ts": iso(day0 + 18000), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], day0 - 10, 0)
    # 例外日 2023-11-15：00:00–05:00（UTC）开放，其余闭馆（weekly 为空）
    resp = client.put(f"/exhibits/{eid}/calendar", json={
        "timezone": "UTC",
        "weekly": [],
        "exceptions": [{"date": "2023-11-15",
                        "windows": [{"start": "00:00", "end": "05:00"}]}],
    })
    assert resp.status_code == 200, resp.text

    # 窗口 02:00–05:00：左半 (02:00冲突→03:00) 无起点端点 → 开放但缺测；
    # (03:00→04:00→05:00) 是冲突后的完整有效段 → 覆盖 2h、200 lux·h
    got = dose(client, eid, day0 + 7200, day0 + 18000,
               extra={"calendar": "display"})
    assert got["known_dose_luxh"] == 200.0
    assert got["covered_seconds"] == 7200.0
    assert got["unknown_seconds"] == 3600.0
    assert got["closed_seconds"] == 0.0
    assert (got["covered_seconds"] + got["unknown_seconds"]
            + got["closed_seconds"]) == 3 * 3600

    # 冲突落在闭馆段内：当日 15:00–18:00（UTC）不在开放区间内，
    # 闭馆时长照常统计，跨冲突对也不产生任何覆盖
    send(client, eid, [
        {"id": "e1", "ts": iso(day0 + 15 * 3600), "lux": 100, "source_id": "a"},
        {"id": "e2", "ts": iso(day0 + 16 * 3600), "lux": 100, "source_id": "c"},
        {"id": "f2", "ts": iso(day0 + 16 * 3600), "lux": 900, "source_id": "d"},
        {"id": "e3", "ts": iso(day0 + 17 * 3600), "lux": 100, "source_id": "a"},
    ])
    w0 = day0 + 15 * 3600
    got2 = dose(client, eid, w0, w0 + 3 * 3600,
                extra={"calendar": "display"})
    assert got2["known_dose_luxh"] == 0.0
    assert got2["covered_seconds"] == 0.0
    assert got2["unknown_seconds"] == 0.0
    assert got2["closed_seconds"] == 3 * 3600


def test_multiple_conflicts_split_into_independent_runs(client):
    """连续多个冲突时刻把时间线切成多段，每段独立积分 / 统计。"""
    eid = make_exhibit(client, interval=1800.0)
    send(client, eid, [
        {"id": "a0", "ts": iso(BASE), "lux": 10, "source_id": "a"},
        {"id": "c1", "ts": iso(BASE + 900), "lux": 10, "source_id": "c"},
        {"id": "d1", "ts": iso(BASE + 900), "lux": 90, "source_id": "d"},
        {"id": "c2", "ts": iso(BASE + 1800), "lux": 20, "source_id": "c"},
        {"id": "d2", "ts": iso(BASE + 1800), "lux": 80, "source_id": "d"},
        {"id": "c3", "ts": iso(BASE + 2700), "lux": 30, "source_id": "c"},
        {"id": "d3", "ts": iso(BASE + 2700), "lux": 70, "source_id": "d"},
        {"id": "a4", "ts": iso(BASE + 3600), "lux": 40, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    got = dose(client, eid, BASE, BASE + 3600)
    # 每个权威点都是独立单端点段：梯形段全部跨冲突 → 无有效积分
    assert got["known_dose_luxh"] == 0.0
    assert got["covered_seconds"] == 0.0
    assert got["unknown_seconds"] == 3600.0
    assert got["reading_count"] == 2  # 仅窗口两端的权威点
    tl = timeline(client, eid, BASE, BASE + 3600)
    assert [e["status"] for e in tl["entries"]] == \
        ["ok", "conflict", "conflict", "conflict", "ok"]


# ---------- 诊断连续链在仲裁冲突两侧重置（回归） ----------

def test_diagnose_chains_reset_at_arbitrated_conflict_without_event(client):
    """仲裁冲突时刻：不生成 timestamp_conflict，且 gap / jump / stuck 链
    在两侧重置，绝不把两侧读数直接相连。"""
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        # 前侧：平稳达 stuck 门槛（0/1800/3600 各一条，照度差 ≤ 1）
        {"id": "l0", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "l1", "ts": iso(BASE + 1800), "lux": 100, "source_id": "a"},
        {"id": "l2", "ts": iso(BASE + 3600), "lux": 101, "source_id": "a"},
        # 中间冲突时刻（最高级并列、异值：c/d 未列入优先级，并列最低）
        {"id": "cx", "ts": iso(BASE + 5400), "lux": 100, "source_id": "c"},
        {"id": "cy", "ts": iso(BASE + 5400), "lux": 500, "source_id": "d"},
        # 后侧：与前侧直接相连会构成大 jump（101 → 900）与长 stuck 假象
        {"id": "r0", "ts": iso(BASE + 7200), "lux": 900, "source_id": "a"},
        {"id": "r1", "ts": iso(BASE + 9000), "lux": 900, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE + 10800), "lux": 899, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)

    events = diagnose(client, eid, BASE, BASE + 10800,
                      max_rate=1.0, stuck_tol=2.0, min_stuck=1800.0)["events"]
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    # 仲裁冲突不产生 timestamp_conflict
    assert "timestamp_conflict" not in by_type
    # 不形成任何跨冲突的 gap / jump
    assert "gap" not in by_type
    assert "jump" not in by_type
    # stuck 在两侧各自成链，均不跨越 BASE+5400
    stuck = by_type.get("stuck", [])
    assert len(stuck) == 2
    assert stuck[0]["reading_ids"] == ["l0", "l1", "l2"]
    assert stuck[1]["reading_ids"] == ["r0", "r1", "r2"]

    # 对比：未配置优先级（平权）时同一数据仍报告 timestamp_conflict
    eid2 = make_exhibit(client, interval=3600.0)
    send(client, eid2, [
        {"id": "x1", "ts": iso(BASE), "lux": 100},
        {"id": "x2", "ts": iso(BASE + 5400), "lux": 100},
        {"id": "x3", "ts": iso(BASE + 5400), "lux": 500},
        {"id": "x4", "ts": iso(BASE + 7200), "lux": 900},
    ])
    flat = diagnose(client, eid2, BASE, BASE + 7200)["events"]
    assert [e["type"] for e in flat if e["type"] == "timestamp_conflict"]


# ---------- 诊断基于权威时间线 ----------

def test_diagnose_uses_authoritative_timeline(client):
    """低优先级来源的尖峰被仲裁丢弃，不产生 jump；未配置时则是时间冲突。"""
    eid = make_exhibit(client, interval=3600.0)
    batch = []
    for i in range(3):
        batch.append({"id": f"a{i}", "ts": iso(BASE + 1800 * i), "lux": 100,
                      "source_id": "a"})
        batch.append({"id": f"b{i}", "ts": iso(BASE + 1800 * i), "lux": 999,
                      "source_id": "b"})
    send(client, eid, batch)

    # 未配置优先级：同时刻异值 → 每个时刻都是 timestamp_conflict
    events = diagnose(client, eid, BASE, BASE + 7200)["events"]
    assert len([e for e in events if e["type"] == "timestamp_conflict"]) == 3

    # 配置 a 优先：b 的尖峰被丢弃，时间线平稳，无任何事件
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    events = diagnose(client, eid, BASE, BASE + 7200)["events"]
    assert events == []


# ---------- 日历 + 仲裁 ----------

def test_calendar_dose_with_arbitration(client):
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "b"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    # 每周 7 天全天开放
    resp = client.put(f"/exhibits/{eid}/calendar", json={
        "timezone": "Asia/Shanghai",
        "weekly": [{"weekday": d, "start": "00:00", "end": "00:00"} for d in range(1, 8)],
    })
    assert resp.status_code == 200, resp.text
    got = dose(client, eid, BASE, BASE + 3600, extra={"calendar": "display"})
    assert got["known_dose_luxh"] == 100.0
    assert got["closed_seconds"] == 0.0


# ---------- 批次更正与仲裁 ----------

def test_correction_voids_sourced_readings_and_recomputes(client):
    """作废高优先级来源的批次后，剩余时间线按低优先级来源重算。"""
    eid = make_exhibit(client, interval=3600.0)
    out1 = send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    send(client, eid, [
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "b"},
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 500, "source_id": "b"},
    ])
    put_priority(client, eid, ["a", "b"], BASE - 10, 0)
    assert dose(client, eid, BASE, BASE + 3600)["known_dose_luxh"] == 100.0

    resp = client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": out1["receipt"]["receipt_id"],
        "request_key": "void-a-batch",
        "reason": "a 传感器漂移，批次作废",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["voided_count"] == 2
    # a 的读数退出时间线后，b 成为权威来源
    assert dose(client, eid, BASE, BASE + 3600)["known_dose_luxh"] == 500.0
    tl = timeline(client, eid, BASE, BASE + 3600)
    assert [e["source_ids"] for e in tl["entries"]] == [["b"], ["b"]]


# ---------- 别名访问与归并合并配置 ----------

def test_alias_access_hits_canonical_config(client):
    make_exhibit(client, eid="EX-S", name="重复档案")
    make_exhibit(client, eid="EX-T", name="规范档案")
    put_priority(client, "EX-T", ["a", "b"], BASE, 0)
    merge_commit(client, "EX-S", "EX-T")

    # 别名读配置：命中规范档案
    got = client.get("/exhibits/EX-S/source-priority")
    assert got.status_code == 200
    body = got.json()
    assert body["requested_id"] == "EX-S"
    assert body["canonical_id"] == "EX-T"
    assert body["current_version"] == 1

    # 别名发版本：写入规范档案
    v2 = put_priority(client, "EX-S", ["b", "a"], BASE + 3600, 1)
    assert v2["canonical_id"] == "EX-T" and v2["version"] == 2

    # 别名查时间线：命中规范档案
    send(client, "EX-S", [{"id": "r1", "ts": iso(BASE + 7200), "lux": 42,
                           "source_id": "b"}])
    tl = timeline(client, "EX-S", BASE + 7200, BASE + 7201)
    assert tl["canonical_id"] == "EX-T"
    assert tl["entries"][0]["lux"] == 42.0
    assert tl["entries"][0]["priority_version"] == 2


def test_merge_moves_source_config_when_target_has_none(client):
    make_exhibit(client, eid="EX-S")
    make_exhibit(client, eid="EX-T")
    put_priority(client, "EX-S", ["a", "b"], BASE, 0)

    preview = client.post("/merges/preview",
                          json={"source_id": "EX-S", "target_id": "EX-T"})
    assert preview.status_code == 200
    body = preview.json()
    assert body["source_priority_conflict"] is False
    assert body["ready_to_commit"] is True
    assert "source_priority" in body["metadata"]  # 差异明示：来源有、目标无

    commit = client.post("/merges/commit",
                         json={"token": body["token"], "resolutions": []})
    assert commit.status_code == 200, commit.text
    got = client.get("/exhibits/EX-T/source-priority").json()
    assert got["current_version"] == 1
    assert got["versions"][0]["priorities"] == ["a", "b"]


def test_merge_identical_configs_commit_ok(client):
    make_exhibit(client, eid="EX-S")
    make_exhibit(client, eid="EX-T")
    put_priority(client, "EX-S", ["a", "b"], BASE, 0)
    put_priority(client, "EX-T", ["a", "b"], BASE, 0)

    preview = client.post("/merges/preview",
                          json={"source_id": "EX-S", "target_id": "EX-T"})
    body = preview.json()
    assert body["source_priority_conflict"] is False
    assert "source_priority" not in body["metadata"]  # 配置一致，无差异
    commit = client.post("/merges/commit",
                         json={"token": body["token"], "resolutions": []})
    assert commit.status_code == 200, commit.text
    # 目标保留一份配置
    got = client.get("/exhibits/EX-T/source-priority").json()
    assert got["current_version"] == 1


def test_merge_conflicting_configs_rejected(client):
    """双方都有配置且不一致：冲突不得静默覆盖，归并 409。"""
    make_exhibit(client, eid="EX-S")
    make_exhibit(client, eid="EX-T")
    put_priority(client, "EX-S", ["a", "b"], BASE, 0)
    put_priority(client, "EX-T", ["b", "a"], BASE, 0)

    preview = client.post("/merges/preview",
                          json={"source_id": "EX-S", "target_id": "EX-T"})
    body = preview.json()
    assert body["source_priority_conflict"] is True
    assert body["ready_to_commit"] is False
    diff = body["metadata"]["source_priority"]
    assert diff["source"][0]["priorities"] == ["a", "b"]
    assert diff["target"][0]["priorities"] == ["b", "a"]

    commit = client.post("/merges/commit",
                         json={"token": body["token"], "resolutions": []})
    assert commit.status_code == 409
    # 双方配置原样保留，未被覆盖
    assert client.get("/exhibits/EX-S/source-priority").json()["current_version"] == 1
    assert (client.get("/exhibits/EX-T/source-priority")
            .json()["versions"][0]["priorities"] == ["b", "a"])


def test_merge_preview_token_invalidated_by_config_change(client):
    """预演后任一侧发布新优先级版本：版本令牌失效，提交 409。"""
    make_exhibit(client, eid="EX-S")
    make_exhibit(client, eid="EX-T")
    preview = client.post("/merges/preview",
                          json={"source_id": "EX-S", "target_id": "EX-T"})
    token = preview.json()["token"]
    put_priority(client, "EX-T", ["a"], BASE, 0)
    commit = client.post("/merges/commit", json={"token": token, "resolutions": []})
    assert commit.status_code == 409


def test_merge_preserves_reading_source_id(client):
    """归并搬迁读数时保留来源标识，归并后仲裁继续生效。"""
    make_exhibit(client, eid="EX-S")
    make_exhibit(client, eid="EX-T")
    send(client, "EX-S", [
        {"id": "r1", "ts": iso(BASE), "lux": 100, "source_id": "a"},
        {"id": "r2", "ts": iso(BASE), "lux": 500, "source_id": "b"},
        {"id": "r3", "ts": iso(BASE + 3600), "lux": 100, "source_id": "a"},
    ])
    put_priority(client, "EX-S", ["a", "b"], BASE - 10, 0)
    merge_commit(client, "EX-S", "EX-T")

    rows = {r["id"]: r for r in client.get("/exhibits/EX-T/readings").json()["readings"]}
    assert rows["r2"]["source_id"] == "b"
    # 归并后配置已搬迁：剂量按权威时间线核算
    assert dose(client, "EX-T", BASE, BASE + 3600)["known_dose_luxh"] == 100.0
