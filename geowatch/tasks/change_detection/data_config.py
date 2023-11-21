
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"

    def get_data_config(self, data_name, is_rectify=False):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = './data/LEVIR-CD/ALL/'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            # raise TypeError('%s has not defined' % data_name)
            self.root_dir = './' + data_name

        if not is_rectify:
            self.org_prefix = "org_"
            self.depth_prefix = 'segW_'
        else:
            self.org_prefix = "rect_"
            self.depth_prefix = 'segW1_'

        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
