class SynonymFinderParams:
    def __init__(self,
                 synonym_count: int = 20,
                 depth: int = 1,
                 depth_weight_multiplier: float = 0.6):
        self.synonym_count = synonym_count
        self.depth = depth
        self.depth_weight_multiplier = depth_weight_multiplier

    def __repr__(self) -> str:
        return f"[synonym_count: {self.synonym_count}; "\
               f"depth: {self.depth}; " \
               f"depth_weight_multiplier: {self.depth_weight_multiplier}]"
