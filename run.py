from active_learning.jobs.job import search_file
from active_learning.maker.maker import MaceMaker,MdMaker

from jobflow import job,Flow
from jobflow import run_locally

mace = MaceMaker(4).make()
f = search_file('active_learning/MACE_models')
md = MdMaker().make(f.output)
flow = Flow(mace + [f] + md)

run_locally(flow)
