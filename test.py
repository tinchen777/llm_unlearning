# import torch
# print("allocated", torch.cuda.memory_allocated()/1e9)   # PyTorch 真正用的
# print("reserved ", torch.cuda.memory_reserved()/1e9)     # PyTorch 占住的(含缓存)


import logging
from rich.logging import RichHandler

from cobra_log import 

logging.basicConfig(
    level="INFO",
    handlers=[RichHandler()]
)

log = logging.getLogger("rich")

log.info("Hello")
log.warning("Something happened")
log.error("Error!")

