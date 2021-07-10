def main():
    """
    Export pip requirements in a conda env file to a requirements.txt file
    """
    import yaml

    fpath = 'conda_env.yml'
    with open(fpath, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    context = []
    found = []
    for dep in data['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            for line in dep['pip']:
                if line.startswith('--'):
                    context.append(line)
                else:
                    found.append(' '.join([line] + context))

    text = '\n'.join(found)
    print(text)
    #with open('requirements.txt', 'w') as file:
    #    file.write(text)


if __name__ == '__main__':
    main()
