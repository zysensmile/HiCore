
from collections import defaultdict
from loguru import logger
from nltk import ngrams

from crslab.evaluator.base import BaseEvaluator
from crslab.evaluator.utils import nice_report
from .metrics import *


class StandardEvaluator(BaseEvaluator):
    """The evaluator for all kind of model(recommender, conversation, policy)
    
    Args:
        rec_metrics: the metrics to evaluate recommender model, including hit@K, ndcg@K and mrr@K
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training
    """

    def __init__(self, language, file_path=None):
        super(StandardEvaluator, self).__init__()
        self.file_path = file_path
        self.rankfile = []
        # rec
        self.rec_metrics = Metrics()
        # gen
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        # optim
        self.optim_metrics = Metrics()

    def rec_evaluate(self, ranks, label):
        self.rankfile += ranks[:50]
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"hit@{k}", HitMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"ndcg@{k}", NDCGMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"mrr@{k}", MRRMetric.compute(ranks, label, k))
        for k in [5, 10, 15, 20]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"cov@{k}", CovMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"iso@{k}", IsoMetric.compute(ranks, label, k))

    def gen_evaluate(self, hyp, refs, seq=None):
        if hyp:
            self.gen_metrics.add("f1", F1Metric.compute(hyp, refs))

            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
                for token in ngrams(seq, k):
                    self.dist_set[f"dist@{k}"].add(token)
            self.dist_cnt += 1

    def report(self, epoch=-1, mode='test'):
        if self.file_path is not None and len(self.rankfile) > 0:
            with open(self.file_path, "w", encoding="utf-8") as file:
                for rank in self.rankfile:
                    file.write(f"{rank} ")
        self.rankfile = []
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / self.dist_cnt))
        reports = [self.rec_metrics.report(), self.gen_metrics.report(), self.optim_metrics.report()]
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        # rec
        self.rec_metrics.clear()
        # conv
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        # optim
        self.optim_metrics.clear()
