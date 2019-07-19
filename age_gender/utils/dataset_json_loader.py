class DatasetJsonLoader:
    def __init__(self, config, dataset):
        self.config = config
        self.init_dataset = dataset
        self.ages = config['ages']
        self.weights = config['weights']

    def get_dataset(self):
        balanced_dataset = []
        for (ind, max_age), weight in zip(enumerate(self.ages), self.weights):
            min_age = 0 if ind == 0 else self.ages[ind - 1]
            age_dataset = self.get_age_class(min_age, max_age, self.init_dataset)
            balanced_class = self.repeat_age_class(weight, age_dataset)
            balanced_dataset += balanced_class
        return balanced_dataset

    @staticmethod
    def get_age_class(min_age, max_age, dataset):
        return list(filter(lambda item: min_age < item['age'] <= max_age, dataset))

    def repeat_age_class(self, weight, age_dataset):
        repeat_number = int(weight)
        subset_len = int(len(age_dataset) * (weight % 1))
        res = list()
        res.extend(age_dataset)
        for i in range(repeat_number - 1):
            res.extend(age_dataset)
        for item in age_dataset[:subset_len]:
            res += [item]
        return res

    @staticmethod
    def get_weights(dataset, ages, target):
        weights = list()
        target_dataset = DatasetJsonLoader.get_age_class(target[0], target[1], dataset)
        for ind, max_age in enumerate(ages):
            min_age = 0 if ind == 0 else ages[ind - 1]
            age_dataset = DatasetJsonLoader.get_age_class(min_age, max_age, dataset)
            weights.append(len(target_dataset) / len(age_dataset))
        return weights