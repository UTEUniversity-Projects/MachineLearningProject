import torch
import torch.nn as nn
import torch.nn.functional as F

class Net01(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv7x7 = nn.Conv2d(3, 32, 7, stride=1, padding=3)

        self.conv3x3_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3x3_2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3x3_3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv3x3_4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3x3_5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)  # 1 MaxPooling

        self.conv3x3_6 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3x3_7 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.conv3x3_8 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3x3_9 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.conv3x3_10 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3x3_11 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv3x3_12 = nn.Conv2d(768, 1024, 3, stride=2, padding=1)

        self.avgpool1 = nn.AvgPool2d(2,2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1) 

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv7x7(x)

        # nhánh 1
        x1 = self.conv3x3_1(x)
        x1 = self.conv3x3_2(x1)
        x1 = self.conv3x3_3(x1)
        x1 = self.conv3x3_4(x1)
        x1 = self.conv3x3_5(x1)
        x1 = self.maxpool(x1)  # MaxPooling

        # nhánh 2
        x2 = self.conv3x3_6(x)
        x2_1 = self.conv3x3_7(x2)
        x2_2 = self.conv3x3_8(x2)

        # Cộng tensors 1
        x3 = F.relu(x2_1 + x2_2)
    

        #torch.cat 1
        x3 = F.relu(torch.cat([x2_1, x3], dim=1))
        x3 = self.conv3x3_10(x3)

        # Adaptive pooling
        x2_2 = self.conv3x3_9(x2_2)
        # Cộng tensors 1
        x4 = F.relu(x2_2 + x3)

        x4 = self.conv3x3_11(x4)
        x4 = self.avgpool1(x4)


        #torch.cat 2
        x5 = F.relu(torch.cat([x1, x4], dim=1))

        x5 = self.conv3x3_12(x5)
        x5 = self.avgpool2(x5)

        x5 = x5.view(-1, 1024)

        # 2 fully connected layer
        x5 = self.fc1(x5)
        x5 = self.fc2(x5)

        return x5

class Net02(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 7, stride=1, padding=3)

        self.conv_2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(64, 96, 5, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(96, 128, 5, stride=1, padding=2)

        self.conv_5 = nn.Conv2d(128, 256, 5, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(256, 512, 5, stride=1, padding=1)
        self.conv_test_1 = nn.Conv2d(512, 512, 5, 1, 0)
        self.conv_7 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv_8 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv_9 = nn.Conv2d(128, 256, 5, stride=2, padding=1)
        self.conv_10 = nn.Conv2d(128, 192, 5, stride=2, padding=2)
        self.conv_11 = nn.Conv2d(192, 256, 5, stride=1, padding=2)

        self.conv_12 = nn.Conv2d(512, 512, 5, stride=2, padding=2)

        self.conv_13 = nn.Conv2d(256, 256, 5, stride=2, padding=1)

        self.conv_14 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_1(x)

        x1 = self.conv_2(x)
        x1 = self.maxpool1(x1)
        x1 = self.conv_3(x1)
        x1 = self.conv_4(x1)

        x1 = self.conv_5(x1)
        x1 = self.conv_6(x1)
        x1 = self.conv_test_1(x1)

        x2 = self.conv_7(x)
        x2_1 = self.conv_14(x2)
        x2_2 = F.relu(self.conv_8(x2))
        x3 = F.relu(x2_1 + x2_2)
        x3 = self.conv_10(x3)
        x3 = self.conv_11(x3)

        x3 = self.conv_13(x3)

        x2_2 = self.maxpool2(x2_2)
        x2_2 = self.conv_9(x2_2)
        x4 = F.relu(torch.cat([x2_2, x3], dim=1))
        x4 = self.conv_12(x4)

        x5 = F.relu(x1 + x4)
    
        x5 = self.avgpool(x5)
        x5 = x5.view(-1, 512)
        x5 = self.fc1(x5)

        return x5

class Net03(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Tầng tiền xử lý bắt buộc
        self.conv7x7 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)

        # Các convolution 3x3
        self.conv3x3 = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for in_channels, out_channels in [
                (32, 32), (32, 64), (64, 128), (64, 128), (32, 64),
                (64, 128), (64, 128), (256, 318), (318, 256), (318, 256)
            ]]
        )

        # Các convolution 5x5
        self.conv5x5 = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
            for in_channels, out_channels in [
                (32, 32), (32, 64), (64, 128), (64, 128), (32, 64),
                (64, 128), (64, 128), (256, 318), (318, 256), (318, 256)
            ]]
        )

        # Adaptive Avg Pooling
        self.adap_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output cho num_classes lớp

    def forward(self, x):
        # Tầng tiền xử lý
        x = self.conv7x7(x)  # [B, 32, 224, 224]

        # Branch 1 -----------------------------------------------------------------------
        branch1 = self.conv3x3[0](x)  # [B, 32, 224, 224]

        #Sub_branch 1_1
        branch1_1 = self.conv3x3[1](branch1)  # [B, 64, 224, 224]
        # Max pool lần 1
        branch1_1 = F.max_pool2d(branch1_1, kernel_size=2, stride=2) # [B, 64, 112, 112]
        # Sub branch 1_2
        branch1_2 = self.conv5x5[1](branch1)  # [B, 64, 224, 224]
        # Avg pool lần 1
        branch1_2 = F.avg_pool2d(branch1_2, kernel_size=2, stride=2) # [B, 64, 112, 112]

        # Branch 1_1
        branch1_1_1 = self.conv3x3[2](branch1_1)  # [B,  128, 112, 112]
        branch1_1_2 = self.conv5x5[2](branch1_1)  # [B, 128, 112, 112]
        # Branch 1_2
        branch1_2_1 = self.conv3x3[3](branch1_2)  # [B, 128, 112, 112]
        branch1_2_2 = self.conv5x5[3](branch1_2)  # [B, 128, 112, 112]

        # Cộng lần 1
        b1_1_plus = branch1_1_1 + branch1_2_1  # [B, 128, 112, 112]
        # Cộng lần 2
        b1_2_plus = branch1_1_2 + branch1_2_2  # [B, 128, 112, 112]

        # Cat lần 1
        b1_cat = torch.cat((b1_1_plus, b1_2_plus), dim=1) # [B, 256, 112, 112]
        b1_cat = self.conv3x3[7](b1_cat) # [B, 318, 112, 112]
        # Max pool lần 2
        b1_cat = F.max_pool2d(b1_cat, kernel_size=4, stride=3) # [B, 318, 37, 37]

        x1 = self.conv3x3[8](b1_cat) # [B, 256, 37, 37]
        x2 = self.conv5x5[8](b1_cat) # [B, 256, 37, 37]

        # Cat lần 2
        b1_cat = torch.cat((x1, x2), dim=1) # [B, 512, 37, 37]

        # Branch 2 -----------------------------------------------------------------------
        branch2 = self.conv5x5[0](x)  # [B, 32, 224, 224]

        # Sub_branch 2_1
        branch2_1 = self.conv3x3[4](branch2)  # [B, 64, 224, 224]
        # Avg pool lần 2
        branch2_1 = F.avg_pool2d(branch2_1, kernel_size=2, stride=2) # [B, 64, 112, 112]
        # Sub_branch 2_2
        branch2_2 = self.conv5x5[4](branch2)  # [B, 64, 224, 224]
        # Max pool lần 3
        branch2_2 = F.max_pool2d(branch2_2, kernel_size=2, stride=2) # [B, 64, 112, 112]

        # Branch 2_1
        branch2_1_1 = self.conv3x3[5](branch2_1)  # [B, 128, 112, 112]
        branch2_1_2 = self.conv5x5[5](branch2_1)  # [B, 128, 112, 112]
        # Branch 2_2
        branch2_2_1 = self.conv3x3[6](branch2_2)  # [B, 128, 112, 112]
        branch2_2_2 = self.conv5x5[6](branch2_2)  # [B, 128, 112, 112]

        # Cộng lần 3
        b2_1_plus = branch2_1_1 + branch2_2_1  # [B, 128, 112, 112]
        # Cộng lần 4
        b2_2_plus = branch2_1_2 + branch2_2_2  # [B, 128, 112, 112]

        # Cat lần 3
        b2_cat = torch.cat((b2_1_plus, b2_2_plus), dim=1) # [B, 256, 112, 112]
        b2_cat = self.conv5x5[7](b2_cat) # [B, 318, 112, 112]
        # Avg pool lần 3
        b2_cat = F.avg_pool2d(b2_cat, kernel_size=4, stride=3) # [B, 318, 37, 37]

        x3 = self.conv3x3[9](b2_cat) # [B, 256, 37, 37]
        x4 = self.conv5x5[9](b2_cat) # [B, 256, 37, 37]

        # Cat lần 4
        b2_cat = torch.cat((x3, x4), dim=1) # [B, 512, 37, 37]

        # Gộp các branch -----------------------------------------------------------------
        # Cat lần 5
        x = F.relu(torch.cat((b1_cat, b2_cat), dim=1))  # [B, 1024, 37, 37]

        # Average Pooling
        x = self.adap_avgpool(x)  # [B, 1024, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [B, 1024]

        # Fully connected
        x = F.relu(self.fc1(x))  # [B, 512]
        x = F.relu(self.fc2(x))  # [B, 256]
        x = self.fc3(x)  # [B, num_classes]
        return x


# from torchsummary import summary
# net = Net()
# net.to(device)
# summary(net, (3, 224, 224))