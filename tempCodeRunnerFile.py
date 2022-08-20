alog.DontUseNativeDialog | dialog.ShowDirsOnly)
        dialog.setOptions(options)

        def checkLineEdit(path):
            if not path:
                return