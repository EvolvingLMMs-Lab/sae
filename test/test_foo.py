import unittest


class TestFoo(unittest.TestCase):
    def test_foo(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
