# example: 192.168.1.0:0.0
$WSL2IP = (
    Get-NetIPAddress -AddressFamily Ipv4 | 
    Where-Object -FilterScript {
        $_.InterfaceAlias -Eq "vEthernet (WSL)" 
    }
).IPv4Address

Write-Host "Your WSL2 IP address is $WSL2IP"

if  (-not $WSL2IP -match "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
{
    Write-Host "please add your WSL 2 IP address in this script"
    Exit 
}
Set-Variable -name DISPLAY -value "${WSL2IP}:0.0"

docker run --rm -it -e DISPLAY=$DISPLAY uas_docker_env:1.0 