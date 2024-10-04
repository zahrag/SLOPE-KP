
from tqdm import tqdm


def train(configs, dataloader, model, optimizer, camera_optimizer, io_module, start_epoch=0):

    for ep in range(start_epoch, configs.max_epoch):

        print('Epoch {}'.format(ep))

        model.multiplex.datasetCameraPoseDict = {}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc='train', position=0)):

            optimizer.zero_grad()

            loss_dict = model.forward(batch)
            total_loss = loss_dict['total_loss']
            total_loss.backward()

            if configs.learn_multiplex:
                camera_optimizer.zero_grad()
                camera_loss = loss_dict['camera_loss_multiplex']
                camera_loss.backward()

        if (ep + 1) % configs.save_interval == 0:
            io_module.save_model(model, optimizer, ep)
            io_module.save_multiplex(model, camera_optimizer, ep)
            io_module.save_logs(model.multiplex.datasetCameraPoseDict, 'campose', ep)
            io_module.save_logs(model.batch_stats, 'raw', ep)



