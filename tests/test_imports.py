import custom_trainer
import unittest


class TestImports(unittest.TestCase):
    def test_import_schedulers(self):
        self.assertTrue(hasattr(custom_trainer, 'Scheduler'))

    def test_import_trainer(self):
        self.assertTrue(hasattr(custom_trainer, 'Trainer'))

    def test_import_utils(self):
        self.assertTrue(hasattr(custom_trainer, 'DictList'))


if __name__ == '__main__':
    unittest.main()
