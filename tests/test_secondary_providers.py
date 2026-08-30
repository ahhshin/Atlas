from __future__ import annotations

import json
from datetime import UTC, datetime

from world_state.ingest.base import RawPayload
from world_state.ingest.providers.airnow import AirNowProvider
from world_state.ingest.providers.eia import EiaProvider
from world_state.ingest.providers.firms import FirmsProvider
from world_state.ingest.providers.usgs import UsgsProvider

NOW = datetime(2026, 8, 30, 20, tzinfo=UTC)


def payload(document, media_type="application/json"):
    content = document if isinstance(document, bytes) else json.dumps(document).encode()
    return RawPayload("fixture", content, "https://example.test", media_type)


def test_usgs_latest_instantaneous_value_becomes_point_batch():
    document = {
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {
                        "siteName": "River",
                        "siteCode": [{"value": "0123"}],
                        "geoLocation": {
                            "geogLocation": {"latitude": 40, "longitude": -80}
                        },
                    },
                    "variable": {
                        "variableCode": [{"value": "00060"}],
                        "unit": {"unitCode": "ft3/s"},
                    },
                    "values": [
                        {
                            "value": [
                                {
                                    "value": "15.2",
                                    "dateTime": "2026-08-30T19:45:00Z",
                                    "qualifiers": ["P"],
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    }
    record = UsgsProvider({}, {}).normalize([payload(document)], NOW)[0].records[0]
    assert record.variable == "streamflow"
    assert record.value == 15.2
    assert record.station_id == "0123"


def test_firms_csv_becomes_fire_event_collection():
    content = (
        b"latitude,longitude,acq_date,acq_time,bright_ti4,frp,confidence,satellite,daynight\n"
        b"40,-80,2026-08-30,1945,330,12.5,h,N,D\n"
    )
    artifact = FirmsProvider({}, {}).normalize([payload(content, "text/csv")], NOW)[0]
    assert artifact.events.iloc[0].fire_radiative_power_mw == 12.5
    assert artifact.events.iloc[0].geometry["type"] == "Point"


def test_airnow_and_eia_keep_measurement_and_publication_context():
    air = [
        {
            "Parameter": "PM25",
            "Value": 8.2,
            "Latitude": 40,
            "Longitude": -80,
            "UTC": "2026-08-30T19:00:00Z",
            "FullAQSCode": "site",
            "Unit": "UG/M3",
        }
    ]
    air_record = AirNowProvider({}, {}).normalize([payload(air)], NOW)[0].records[0]
    assert air_record.variable == "pm2_5"
    eia = {
        "response": {
            "data": [
                {
                    "period": "2026-08-30T19",
                    "respondent": "PJM",
                    "respondent-name": "PJM Interconnection",
                    "type": "D",
                    "value": "120000",
                    "value-units": "megawatthours",
                }
            ]
        }
    }
    eia_record = EiaProvider({}, {}).normalize([payload(eia)], NOW)[0].records[0]
    assert eia_record.variable == "electricity_demand"
    assert eia_record.latitude is None
    assert eia_record.station_id == "PJM"
