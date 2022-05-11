#!/usr/bin/python
import csv

import random

from decimal import Decimal as dec

records=167
print("Making %d records\n" % records)

fieldnames = [
    "Latency (ms)",
    "Jitter (ms)",
    "Bit Rate (Mbps)",
    "Packet Loss Rate (%)",
    "Peak Data Rate DL (Gbps)",
    "Peak Data Rate UL (Gbps)",
    "Mobility (km/h)",
    "Reliability (%)",
    "Service Availability (%)",
    "Survival Time (ms)",
    "Experienced Data Rate DL (Mbps)",
    "Experienced Data Rate UL (Mbps)",
    "Interruption Time (ms)",
    "Service"
]
writer = csv.DictWriter(open("database.csv", "w"), fieldnames=fieldnames)

services = [" UHD_Video_Streaming",
            " Immerse_Experience",
            " Vo5G",
            " e_Health",
            " ITS",
            " Surveillance",
            " Connected_Vehicles",
            " Smart_Grid",
            " Industry_Automation"
            ]


def gen_latency(service: str) -> int:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_jitter(service: str) -> float:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_bit_rate(service: str) -> float:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_plr(service: str) -> dec:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_pdr_dl(service: str) -> float:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_pdr_ul(service: str) -> float:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_mobility(service: str) -> int:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_sr(service: str) -> dec:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_availibilty(service: str) -> dec:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_survival_time(service: str) -> int:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_edr_dl(service: str) -> float:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_edr_ul(service: str) -> int:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)

def gen_interruption_time(service: str) -> int:
    if service == services[0]:
        return random.randint(2, 20)
    elif service == services[1]:
        return random.randint(7, 15)
    elif service == services[2]:
        return random.randint(5, 50)
    elif service == services[3]:
        return random.randint(1, 10)
    elif service == services[4]:
        return random.randint(10, 100)
    elif service == services[5]:
        return random.randint(20, 150)
    elif service == services[6]:
        return random.randint(3, 100)
    elif service == services[7]:
        return random.randint(1, 50)
    elif service == services[8]:
        return random.randint(10, 50)
    else:
        # weir traffic
        return random.randint(0, 1)


writer.writerow(dict(zip(fieldnames, fieldnames)))
for i in range(0, records):
    _service = random.choice(services)
    writer.writerow(dict([
        ("Latency (ms)", gen_latency(_service)),
         ( "Jitter (ms)", gen_jitter(_service)),
           ("Bit Rate (Mbps)", gen_bit_rate(_service)),
            ("Packet Loss Rate (%)", gen_plr(_service)),
             ("Peak Data Rate DL (Gbps)", gen_pdr_dl(_service)),
              ("Peak Data Rate UL (Gbps)", gen_pdr_ul(_service)),
               ("Mobility (km/h)", gen_mobility(_service)),
                ("Reliability (%)", gen_sr(_service)),
                 ("Service Availability (%)", gen_availibilty(_service)),
                  ("Survival Time (ms)", gen_survival_time(_service)),
                   ("Experienced Data Rate DL (Mbps)", gen_edr_dl(_service)),
                    ("Experienced Data Rate UL (Mbps)", gen_edr_ul(_service)),
                     ("Interruption Time (ms)", gen_interruption_time(_service)),
                      ("Service", _service)
      ])

  )
